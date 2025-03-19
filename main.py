import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pydicom
from stl import mesh
from tqdm import tqdm
import gc
from skimage.measure import marching_cubes
from skimage.transform import resize  # Для изменения размера данных, если нужно

app = Flask(__name__)

# Настроим CORS
CORS(app, origins="http://localhost:5173")

# Папки для загрузки и сохранения STL-моделей
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'stl_models'
ALLOWED_EXTENSIONS = {'dcm'}

# Создаем необходимые папки, если их нет
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER


# Проверка расширения файла
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Преобразуем DICOM в STL с использованием Marching Cubes
def dicom_to_stl(dicom_files):
    print("Starting DICOM to STL conversion...")

    # Собираем все срезы в один массив
    slices = []
    for dicom_file in tqdm(dicom_files, desc="Processing DICOM files"):
        dicom_data = pydicom.dcmread(dicom_file)
        if 'PixelData' in dicom_data:
            # Проверяем, что данные 2D
            if len(dicom_data.pixel_array.shape) == 2:
                slices.append(dicom_data.pixel_array)
            else:
                print(f"Skipping file {dicom_file}: not a 2D slice")
        del dicom_data  # Освобождаем память после обработки каждого файла
        gc.collect()

    # Проверяем, что есть хотя бы один срез
    if len(slices) == 0:
        print("No valid 2D slices found")
        return None

    # Проверяем, что все срезы имеют одинаковую размерность
    first_slice_shape = slices[0].shape
    for i, slice_data in enumerate(slices):
        if slice_data.shape != first_slice_shape:
            print(f"Resizing slice {i} to match the first slice")
            slices[i] = resize(slice_data, first_slice_shape)  # Приводим к одной размерности

    # Преобразуем в 3D-массив
    try:
        data = np.stack(slices)  # Объединяем срезы в 3D-массив
        print(f"Data shape after stacking: {data.shape}")  # Должно быть (depth, height, width)
    except Exception as e:
        print(f"Error stacking slices into 3D array: {e}")
        return None

    # Нормализуем данные (если нужно)
    data = data.astype(np.float32)
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # Применяем алгоритм Marching Cubes для создания 3D-модели
    print("Applying Marching Cubes algorithm...")
    try:
        # Используем Marching Cubes для создания вершин и граней
        verts, faces, _, _ = marching_cubes(data, level=0.5)
    except Exception as e:
        print(f"Error during Marching Cubes: {e}")
        return None

    # Создаем STL-модель
    print("Creating STL mesh...")
    mesh_data = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            mesh_data.vectors[i][j] = verts[face[j]]

    # Сохраняем STL
    stl_filename = os.path.join(MODEL_FOLDER, 'converted_model.stl')
    mesh_data.save(stl_filename)

    # Освобождаем память
    del mesh_data, verts, faces, data
    gc.collect()

    return stl_filename


# Маршрут для загрузки файлов
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'}), 400

    files = request.files.getlist('files')
    if len(files) == 0:
        return jsonify({'error': 'No files received'}), 400

    # Сохраняем файлы DICOM на сервере
    dicom_paths = []
    for file in files:
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            try:
                file.save(filepath)
                dicom_paths.append(filepath)
                print(f"File saved to {filepath}")
            except Exception as e:
                return jsonify({'error': f'Error saving file: {str(e)}'}), 500

    # Преобразуем DICOM файлы в STL
    try:
        stl_file = dicom_to_stl(dicom_paths)
        if stl_file:
            return jsonify({'message': 'File successfully uploaded and converted', 'stl_file': stl_file}), 200
        else:
            return jsonify({'error': 'Failed to convert DICOM to STL'}), 500
    except Exception as e:
        return jsonify({'error': f'Error during DICOM to STL conversion: {str(e)}'}), 500


# Маршрут для получения STL-файла по URL
@app.route('/stl_models/<filename>', methods=['GET'])
def get_stl_model(filename):
    return send_from_directory(app.config['MODEL_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)