import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pydicom
import open3d as o3d  # Используем Open3D для сглаживания
from tqdm import tqdm
import gc
from skimage.measure import marching_cubes
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from stl import mesh as np_mesh  # Используем numpy-stl вместо stl

app = Flask(__name__)

CORS(app)  # Разрешает CORS для всех источников

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'stl_models'
ALLOWED_EXTENSIONS = {'dcm'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def smooth_mesh_open3d(o3d_mesh, iterations=10):
    """Сглаживание с использованием Open3D (Laplacian или Taubin filter)"""
    o3d_mesh = o3d_mesh.filter_smooth_laplacian(number_of_iterations=iterations)
    return o3d_mesh

def clear_upload_folder():
    """Очищает папку uploads после завершения обработки"""
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def dicom_to_stl(dicom_files):
    print("Starting DICOM to STL conversion...")

    slices = []
    for dicom_file in tqdm(dicom_files, desc="Processing DICOM files"):
        dicom_data = pydicom.dcmread(dicom_file)
        if 'PixelData' in dicom_data:
            slice_data = dicom_data.pixel_array.astype(np.int16)

            # Проверяем, есть ли корректные HU значения
            if hasattr(dicom_data, 'RescaleSlope') and hasattr(dicom_data, 'RescaleIntercept'):
                slice_data = slice_data * dicom_data.RescaleSlope + dicom_data.RescaleIntercept

            # Фильтруем кости: HU > 200
            slice_data[slice_data < 200] = 0

            slices.append(slice_data)

        del dicom_data
        gc.collect()

    if len(slices) == 0:
        print("No valid 2D slices found")
        return None

    # Приводим к единому размеру
    first_slice_shape = slices[0].shape
    for i in range(len(slices)):
        if slices[i].shape != first_slice_shape:
            slices[i] = resize(slices[i], first_slice_shape, anti_aliasing=True)

    data = np.stack(slices)

    # Сглаживание для уменьшения артефактов
    data = gaussian_filter(data, sigma=1)

    # Нормализация (только костные ткани)
    min_hu, max_hu = 200, 1000
    data = np.clip(data, min_hu, max_hu)
    data = (data - min_hu) / (max_hu - min_hu)

    # Выбор оптимального порога для Marching Cubes
    threshold = np.percentile(data[data > 0], 50)  # Среднее значение по костям

    print("Applying Marching Cubes...")
    try:
        verts, faces, _, _ = marching_cubes(data, level=threshold)
    except Exception as e:
        print(f"Error during Marching Cubes: {e}")
        return None

    print("Creating Open3D mesh...")
    # Преобразуем вершины и треугольники в формат Open3D
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Сглаживаем модель с помощью Open3D
    print("Smoothing mesh...")
    o3d_mesh = smooth_mesh_open3d(o3d_mesh, iterations=10)

    # Извлекаем новые вершины и треугольники
    smooth_verts = np.asarray(o3d_mesh.vertices)
    smooth_faces = np.asarray(o3d_mesh.triangles)

    print("Saving STL model...")

    # Сохраняем результат в новый STL файл с использованием numpy-stl
    mesh_data = np_mesh.Mesh(np.zeros(smooth_faces.shape[0], dtype=np_mesh.Mesh.dtype))
    for i, face in enumerate(smooth_faces):
        for j in range(3):
            mesh_data.vectors[i][j] = smooth_verts[face[j]]

    stl_filename = os.path.join(MODEL_FOLDER, 'converted_model_smoothed.stl')
    mesh_data.save(stl_filename)

    # Очищаем папку uploads после сохранения STL
    clear_upload_folder()

    del mesh_data, smooth_verts, smooth_faces, data
    gc.collect()

    return stl_filename

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'}), 400

    files = request.files.getlist('files')
    if len(files) == 0:
        return jsonify({'error': 'No files received'}), 400

    dicom_paths = []
    for file in files:
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            try:
                file.save(filepath)
                dicom_paths.append(filepath)
            except Exception as e:
                return jsonify({'error': f'Error saving file: {str(e)}'}), 500

    try:
        stl_file = dicom_to_stl(dicom_paths)
        if stl_file:
            return jsonify({'message': 'File successfully uploaded and converted', 'stl_file': stl_file}), 200
        else:
            return jsonify({'error': 'Failed to convert DICOM to STL'}), 500
    except Exception as e:
        return jsonify({'error': f'Error during DICOM to STL conversion: {str(e)}'}), 500

@app.route('/stl_models/<filename>', methods=['GET'])
def get_stl_model(filename):
    return send_from_directory(app.config['MODEL_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
