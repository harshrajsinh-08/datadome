import requests
from flask_cors import CORS
from flask import Flask, request, render_template, send_file, jsonify, session

import os
import json
import pandas as pd

from utils.profile_report import profile_report
from utils.categorize_columns import cat_c
import utils.global_store as global_store
from utils.secret_key import generate_secret_key

from pre_processing.models import classification_standard, regression_standard, train_predict_regression, visualize_results, visualize_classification_results, train_predict_classification
from pre_processing.image_main import process_image_dataset, get_image_processing_summary
from pre_processing.graph_main import process_graph_dataset, get_graph_processing_summary
from pre_processing.advanced_graph_main import process_advanced_graph_dataset, get_advanced_processing_summary
import numpy as np

app = Flask(__name__)
CORS(app)  

UPLOAD_FOLDER = os.path.join(os.getcwd(), "app", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config['SESSION_PERMANENT'] = False

app.secret_key = generate_secret_key()

def convert_numpy_types(obj):
    """Convert NumPy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/save", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file part", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], "user_data.csv")
    file.save(file_path)

    return render_template("profile.html")

@app.route("/generate", methods=["POST"])
def generateFile():
    profile_report("app/uploads/user_data.csv")
    return send_file("../output/profile_report.html", as_attachment=False)

@app.route("/attribute_cleaning")
def attribute_cleaning():
    df = pd.read_csv("app/uploads/user_data.csv")

    categorized = cat_c(df)
    categorical_data = {}
    for col in categorized["categorical"]:
        categorical_data[col] = df[col].dropna().unique().tolist()
    return render_template("attribute_cleaning.html", 
                           columns = categorical_data,
                           numeric=categorized["numeric"], 
                           categorical=categorized["categorical"], 
                           datetime=categorized["datetime"], 
                           categorical_data=categorical_data)

@app.route("/models")
def models():
    return render_template("models.html")

@app.route("/image_processing")
def image_processing():
    return render_template("image_processing.html")

@app.route("/graph_processing")
def graph_processing():
    return render_template("graph_processing.html")

@app.route("/advanced_graph_processing")
def advanced_graph_processing():
    return render_template("advanced_graph_processing.html")

@app.route("/process_images", methods=["POST"])
def process_images():
    try:
        data = request.get_json()
        
        # Get parameters from request
        dataset_path = data.get("dataset_path", "app/uploads/images")
        min_resolution = data.get("min_resolution", 32*32)
        max_resolution = data.get("max_resolution", 4096*4096)
        min_blur_threshold = data.get("min_blur_threshold", 100)
        min_brightness = data.get("min_brightness", 10)
        max_brightness = data.get("max_brightness", 245)
        min_contrast = data.get("min_contrast", 10)
        remove_duplicates = data.get("remove_duplicates", True)
        remove_corrupted = data.get("remove_corrupted", True)
        test_corruption = data.get("test_corruption", False)
        enhance_images = data.get("enhance_images", False)
        enhancement_options = data.get("enhancement_options", None)
        train_models = data.get("train_models", False)
        max_images_for_ml = data.get("max_images_for_ml", 500)
        csv_path = data.get("csv_path", None)
        target_column = data.get("target_column", None)
        
        # Add test corrupted images if requested
        if test_corruption:
            print("Adding test corrupted images for verification...")
            test_dir = os.path.join(dataset_path, "test_corrupted")
            os.makedirs(test_dir, exist_ok=True)
            
            # Create test corrupted images
            from PIL import Image
            
            # Truncated file
            with open(os.path.join(test_dir, "test_truncated.jpg"), 'wb') as f:
                f.write(b'\xff\xd8\xff\xe0')  # JPEG header but truncated
            
            # Empty file
            with open(os.path.join(test_dir, "test_empty.jpg"), 'wb') as f:
                pass
            
            # All black image
            black_img = Image.new('RGB', (50, 50), color='black')
            black_img.save(os.path.join(test_dir, "test_all_black.jpg"))
            
            print("Test corrupted images added to dataset")
        
        # Check if dataset path exists
        if not os.path.exists(dataset_path):
            return jsonify({"error": f"Dataset path does not exist: {dataset_path}"}), 400
        
        # Clear previous processing results
        import shutil
        output_dirs_to_clear = [
            "output/cleaned_images",
            "output/enhanced_images", 
            "app/static/image_results"
        ]
        
        for output_dir in output_dirs_to_clear:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
                print(f"Cleared previous results from {output_dir}")
        
        # Clear previous CSV results
        csv_files_to_clear = [
            "output/original_image_analysis.csv",
            "output/cleaned_image_analysis.csv",
            "output/image_cleaning_stats.csv",
            "output/image_enhancement_stats.csv",
            "output/image_quality_improvements.csv"
        ]
        
        for csv_file in csv_files_to_clear:
            if os.path.exists(csv_file):
                os.remove(csv_file)
                print(f"Cleared previous result: {csv_file}")
        
        # Process the image dataset
        result = process_image_dataset(
            dataset_path=dataset_path,
            min_resolution=min_resolution,
            max_resolution=max_resolution,
            min_blur_threshold=min_blur_threshold,
            min_brightness=min_brightness,
            max_brightness=max_brightness,
            min_contrast=min_contrast,
            remove_duplicates=remove_duplicates,
            remove_corrupted=remove_corrupted,
            enhance_images=enhance_images,
            enhancement_options=enhancement_options,
            train_models=train_models,
            max_images_for_ml=max_images_for_ml,
            csv_labels_path=csv_path,
            target_column=target_column
        )
        
        # Handle different return formats (with/without enhancement/models)
        if len(result) == 4:
            original_df, cleaned_df, cleaning_stats, quality_report = result
            enhancement_stats = None
            quality_improvements = None
            model_comparison_results = None
        elif len(result) == 6:
            original_df, cleaned_df, cleaning_stats, quality_report, enhancement_stats, quality_improvements = result
            model_comparison_results = None
        else:
            original_df, cleaned_df, cleaning_stats, quality_report, enhancement_stats, quality_improvements, model_comparison_results = result
        
        if original_df is None:
            return jsonify({"error": "No valid images found in the dataset"}), 400
        
        # Generate summary
        summary = get_image_processing_summary(cleaning_stats, quality_report, enhancement_stats, quality_improvements, model_comparison_results)
        
        # Convert NumPy types to JSON-serializable types
        cleaning_stats_json = convert_numpy_types(cleaning_stats)
        quality_report_json = convert_numpy_types(quality_report)
        
        # Prepare response
        response = {
            "success": True,
            "summary": summary,
            "cleaning_stats": cleaning_stats_json,
            "quality_report": quality_report_json,
            "original_count": len(original_df),
            "cleaned_count": len(cleaned_df) if cleaned_df is not None else 0,
            "visualizations": {
                "main_results": "/static/image_results/image_cleaning_results.png",
                "quality_comparison": "/static/image_results/quality_metrics_comparison.png",
                "sample_original": "/static/image_results/sample_original_images.png",
                "sample_cleaned": "/static/image_results/sample_cleaned_images.png",
                "model_comparison": "/static/image_results/model_performance_comparison.png" if train_models else None
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in image processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 500

@app.route("/upload_images", methods=["POST"])
def upload_images():
    """Handle image dataset upload"""
    try:
        # Create upload directory for images
        image_upload_dir = os.path.join(app.config["UPLOAD_FOLDER"], "images")
        
        # Clear previous uploads - remove entire directory and recreate
        if os.path.exists(image_upload_dir):
            import shutil
            shutil.rmtree(image_upload_dir)
            print(f"Cleared previous uploads from {image_upload_dir}")
        
        os.makedirs(image_upload_dir, exist_ok=True)
        
        uploaded_files = []
        
        # Handle multiple file upload
        if 'files' in request.files:
            files = request.files.getlist('files')
            
            for file in files:
                if file and file.filename:
                    # Check if it's an image file
                    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
                    file_ext = os.path.splitext(file.filename)[1].lower()
                    
                    if file_ext in allowed_extensions:
                        file_path = os.path.join(image_upload_dir, file.filename)
                        file.save(file_path)
                        uploaded_files.append(file.filename)
        
        if uploaded_files:
            return jsonify({
                "success": True,
                "message": f"Uploaded {len(uploaded_files)} images",
                "files": uploaded_files,
                "dataset_path": image_upload_dir
            })
        else:
            return jsonify({"error": "No valid image files uploaded"}), 400
            
    except Exception as e:
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    """Handle CSV file upload for image labels"""
    try:
        if 'csv_file' not in request.files:
            return jsonify({"error": "No CSV file provided"}), 400
        
        file = request.files['csv_file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({"error": "File must be a CSV"}), 400
        
        # Save CSV file
        csv_upload_dir = os.path.join(app.config["UPLOAD_FOLDER"], "csv")
        os.makedirs(csv_upload_dir, exist_ok=True)
        
        # Clear previous CSV files
        for existing_file in os.listdir(csv_upload_dir):
            os.remove(os.path.join(csv_upload_dir, existing_file))
        
        csv_path = os.path.join(csv_upload_dir, "labels.csv")
        file.save(csv_path)
        
        # Read and validate CSV
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        return jsonify({
            "success": True,
            "message": f"CSV uploaded successfully: {file.filename}",
            "csv_path": csv_path,
            "columns": df.columns.tolist(),
            "rows": len(df)
        })
        
    except Exception as e:
        return jsonify({"error": f"CSV upload failed: {str(e)}"}), 500

@app.route("/upload_graph", methods=["POST"])
def upload_graph():
    """Handle graph file upload"""
    try:
        if 'graph_file' not in request.files:
            return jsonify({"error": "No graph file provided"}), 400
        
        file = request.files['graph_file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Check file format
        allowed_extensions = {'.csv', '.txt', '.edgelist', '.gml', '.graphml'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({"error": f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"}), 400
        
        # Save graph file
        graph_upload_dir = os.path.join(app.config["UPLOAD_FOLDER"], "graphs")
        
        # Clear previous graph files
        if os.path.exists(graph_upload_dir):
            import shutil
            shutil.rmtree(graph_upload_dir)
        
        os.makedirs(graph_upload_dir, exist_ok=True)
        
        graph_path = os.path.join(graph_upload_dir, "graph" + file_ext)
        file.save(graph_path)
        
        # Determine format
        if file_ext == '.gml':
            graph_format = 'gml'
        elif file_ext == '.graphml':
            graph_format = 'graphml'
        else:
            graph_format = 'edgelist'
        
        # Quick validation
        try:
            if graph_format == 'edgelist':
                if file_ext == '.csv':
                    df = pd.read_csv(graph_path)
                    edge_count = len(df)
                    node_count = len(set(df.iloc[:, 0].tolist() + df.iloc[:, 1].tolist()))
                else:
                    with open(graph_path, 'r') as f:
                        lines = [line.strip() for line in f if line.strip()]
                    edge_count = len(lines)
                    nodes = set()
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 2:
                            nodes.update(parts[:2])
                    node_count = len(nodes)
            else:
                # For GML/GraphML, use NetworkX to get basic info
                import networkx as nx
                if graph_format == 'gml':
                    temp_graph = nx.read_gml(graph_path)
                else:
                    temp_graph = nx.read_graphml(graph_path)
                node_count = temp_graph.number_of_nodes()
                edge_count = temp_graph.number_of_edges()
        
        except Exception as e:
            return jsonify({"error": f"Invalid graph file format: {str(e)}"}), 400
        
        return jsonify({
            "success": True,
            "message": f"Graph uploaded successfully: {file.filename}",
            "graph_path": graph_path,
            "graph_format": graph_format,
            "nodes": node_count,
            "edges": edge_count
        })
        
    except Exception as e:
        return jsonify({"error": f"Graph upload failed: {str(e)}"}), 500

@app.route("/process_graph", methods=["POST"])
def process_graph():
    """Process graph dataset with cleaning and ML comparison"""
    try:
        data = request.get_json()
        
        # Get parameters
        graph_path = data.get("graph_path")
        graph_format = data.get("graph_format", "edgelist")
        directed = data.get("directed", False)
        weighted = data.get("weighted", False)
        remove_isolated_nodes = data.get("remove_isolated_nodes", True)
        remove_self_loops = data.get("remove_self_loops", True)
        remove_duplicate_edges = data.get("remove_duplicate_edges", True)
        min_degree_threshold = data.get("min_degree_threshold", 1)
        keep_largest_component = data.get("keep_largest_component", True)
        min_component_size = data.get("min_component_size", 5)
        train_models = data.get("train_models", False)
        ml_task_type = data.get("ml_task_type", "degree_classification")
        max_nodes_for_ml = data.get("max_nodes_for_ml", 1000)
        
        if not graph_path or not os.path.exists(graph_path):
            return jsonify({"error": "Graph file not found"}), 400
        
        # Clear previous graph processing results
        import shutil
        output_dirs_to_clear = ["app/static/graph_results"]
        
        for output_dir in output_dirs_to_clear:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
        
        # Process the graph dataset
        original_analysis, cleaned_analysis, cleaning_stats, ml_comparison_results = process_graph_dataset(
            graph_file_path=graph_path,
            graph_format=graph_format,
            directed=directed,
            weighted=weighted,
            remove_isolated_nodes=remove_isolated_nodes,
            remove_self_loops=remove_self_loops,
            remove_duplicate_edges=remove_duplicate_edges,
            min_degree_threshold=min_degree_threshold,
            keep_largest_component=keep_largest_component,
            min_component_size=min_component_size,
            train_models=train_models,
            ml_task_type=ml_task_type,
            max_nodes_for_ml=max_nodes_for_ml
        )
        
        if original_analysis is None:
            return jsonify({"error": "Graph processing failed"}), 500
        
        # Generate summary
        summary = get_graph_processing_summary(
            cleaning_stats, original_analysis, cleaned_analysis, ml_comparison_results
        )
        
        # Convert NumPy types for JSON serialization
        cleaning_stats_json = convert_numpy_types(cleaning_stats)
        original_analysis_json = convert_numpy_types(original_analysis)
        cleaned_analysis_json = convert_numpy_types(cleaned_analysis) if cleaned_analysis else None
        
        response = {
            "success": True,
            "summary": summary,
            "cleaning_stats": cleaning_stats_json,
            "original_analysis": original_analysis_json,
            "cleaned_analysis": cleaned_analysis_json,
            "ml_comparison": convert_numpy_types(ml_comparison_results) if ml_comparison_results else None,
            "visualizations": {
                "ml_comparison": "/static/graph_results/graph_ml_comparison.png" if train_models and ml_comparison_results else None
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in graph processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Graph processing failed: {str(e)}"}), 500

@app.route("/process_advanced_graph", methods=["POST"])
def process_advanced_graph():
    """Process graph dataset with advanced techniques including Neo4j integration"""
    try:
        data = request.get_json()
        
        # Get parameters
        graph_path = data.get("graph_path")
        directed = data.get("directed", False)
        weighted = data.get("weighted", False)
        
        # Neo4j parameters
        neo4j_uri = data.get("neo4j_uri", "").strip()
        neo4j_user = data.get("neo4j_user", "").strip()
        neo4j_password = data.get("neo4j_password", "").strip()
        
        # Use Neo4j only if all credentials provided
        if not (neo4j_uri and neo4j_user and neo4j_password):
            neo4j_uri = neo4j_user = neo4j_password = None
        
        # Advanced cleaning parameters
        remove_isolated_nodes = data.get("remove_isolated_nodes", True)
        remove_self_loops = data.get("remove_self_loops", True)
        remove_duplicate_edges = data.get("remove_duplicate_edges", True)
        min_degree_threshold = data.get("min_degree_threshold", 1)
        max_degree_threshold = data.get("max_degree_threshold")
        keep_largest_component = data.get("keep_largest_component", True)
        min_component_size = data.get("min_component_size", 5)
        remove_bridges = data.get("remove_bridges", False)
        anomaly_detection = data.get("anomaly_detection", True)
        preserve_important_nodes = data.get("preserve_important_nodes", True)
        
        # ML parameters
        train_models = data.get("train_models", True)
        ml_task_type = data.get("ml_task_type", "centrality_classification")
        max_nodes_for_ml = data.get("max_nodes_for_ml", 2000)
        use_neo4j_analytics = data.get("use_neo4j_analytics", False)
        export_to_neo4j = data.get("export_to_neo4j", False)
        
        if not graph_path or not os.path.exists(graph_path):
            return jsonify({"error": "Graph file not found"}), 400
        
        # Clear previous graph processing results
        import shutil
        output_dirs_to_clear = ["app/static/graph_results"]
        
        for output_dir in output_dirs_to_clear:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
        
        # Process the graph dataset with advanced techniques
        results = process_advanced_graph_dataset(
            graph_file_path=graph_path,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            directed=directed,
            weighted=weighted,
            remove_isolated_nodes=remove_isolated_nodes,
            remove_self_loops=remove_self_loops,
            remove_duplicate_edges=remove_duplicate_edges,
            min_degree_threshold=min_degree_threshold,
            max_degree_threshold=max_degree_threshold,
            keep_largest_component=keep_largest_component,
            min_component_size=min_component_size,
            remove_bridges=remove_bridges,
            anomaly_detection=anomaly_detection,
            preserve_important_nodes=preserve_important_nodes,
            train_models=train_models,
            ml_task_type=ml_task_type,
            max_nodes_for_ml=max_nodes_for_ml,
            use_neo4j_analytics=use_neo4j_analytics,
            export_to_neo4j=export_to_neo4j
        )
        
        if results is None:
            return jsonify({"error": "Advanced graph processing failed"}), 500
        
        # Generate comprehensive summary
        summary = get_advanced_processing_summary(results)
        
        # Convert NumPy types for JSON serialization
        response = {
            "success": True,
            "summary": summary,
            "original_metrics": convert_numpy_types(results.get("original_metrics", {})),
            "cleaned_metrics": convert_numpy_types(results.get("cleaned_metrics", {})),
            "cleaning_stats": convert_numpy_types(results.get("cleaning_stats", {})),
            "neo4j_analytics": convert_numpy_types(results.get("neo4j_analytics", {})),
            "ml_comparison": convert_numpy_types(results.get("ml_comparison_results", {}).get("comparison", {})) if results.get("ml_comparison_results") else None,
            "exported_files": results.get("exported_files", {}),
            "processing_parameters": results.get("processing_parameters", {}),
            "visualizations": {
                "advanced_ml_comparison": "/static/graph_results/advanced_graph_ml_comparison.png" if train_models and results.get("ml_comparison_results") else None
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in advanced graph processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Advanced graph processing failed: {str(e)}"}), 500

#___________________________________________________________________________________________________________________________________
@app.route("/run_model", methods=["POST"])
def run_model():
    try:
        # Get model name from request
        data = request.get_json()
        model_name = data.get("model_name")
        
        if not model_name:
            return jsonify({"error": "No model name provided"}), 400

        # Define valid models for each task type
        regression_models = ["linear_regression", "decision_tree", "random_forest", "svm", "knn", "gradient_boosting"]
        classification_models = ["logistic_regression", "decision_tree", "random_forest", "svm", "knn", "gradient_boosting"]
        
        # Retrieve result_array from session
        result_array = session.get("result_array")
        print(f"Session result_array: {result_array}")
        
        if not result_array:
            return jsonify({
                "error": "No target column specified. Please enter a target column in the input field above and click on a model button."
            }), 400
        
        if len(result_array) < 3:
            return jsonify({
                "error": "Invalid session data. Please refresh the page and try again."
            }), 400
            
        task_type = result_array[1]
        target_column = result_array[2]
        
        print(f"Task type: {task_type}, Model: {model_name}, Target column: {target_column}")

        # Validate task type
        if task_type not in ["prediction", "classification"]:
            return jsonify({
                "error": f"Invalid task type '{task_type}'. Please select a model from either Regression or Classification section."
            }), 400

        # Validate model selection based on task type
        if task_type == "prediction" and model_name not in regression_models:
            return jsonify({
                "error": f"Model '{model_name}' is not valid for regression tasks. Please select a model from the Regression Models section."
            }), 400
        elif task_type == "classification" and model_name not in classification_models:
            return jsonify({
                "error": f"Model '{model_name}' is not valid for classification tasks. Please select a model from the Classification Models section."
            }), 400

        # Check if data file exists
        data_csv = "app/uploads/user_data.csv"
        if not os.path.exists(data_csv):
            return jsonify({
                "error": "No data file found. Please upload a CSV file first."
            }), 400
        
        # Validate target column exists in dataset
        try:
            df_check = pd.read_csv(data_csv)
            if target_column not in df_check.columns:
                available_columns = ", ".join(df_check.columns.tolist())
                return jsonify({
                    "error": f"Target column '{target_column}' not found in dataset. Available columns: {available_columns}"
                }), 400
        except Exception as e:
            return jsonify({
                "error": f"Failed to read data file: {str(e)}"
            }), 400

        # Train models
        print(f"Starting model training for {model_name}...")
        results = {}
        
        try:
            if task_type == "prediction":
                print("Training regression models...")
                results, y_test, y_test_pred = train_predict_regression(data_csv, model_name, target=target_column)
                un_results, un_y_test, un_y_test_pred = regression_standard(data_csv, model_name, target=target_column)
            elif task_type == "classification":
                print("Training classification models...")
                results, y_test, y_test_pred = train_predict_classification(data_csv, model_name, target=target_column)
                un_results, un_y_test, un_y_test_pred = classification_standard(data_csv, model_name, target=target_column)
            
            print("Model training completed successfully")
            
        except KeyError as e:
            return jsonify({
                "error": f"Column error: {str(e)}. Please check that the target column name is correct."
            }), 400
        except ValueError as e:
            return jsonify({
                "error": f"Value error during training: {str(e)}"
            }), 400
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                "error": f"Model training failed: {str(e)}"
            }), 500
        
        # Generate visualization
        try:
            # Ensure the static directory exists
            static_dir = os.path.join(os.getcwd(), "app", "static")
            os.makedirs(static_dir, exist_ok=True)
            
            # Add model name to the visualization
            model_display_name = model_name.replace('_', ' ').title()
            
            print(f"Generating visualization for {model_display_name}...")
            
            # Use appropriate visualization based on task type
            if task_type == "prediction":
                visualize_results(
                    un_results=un_results,
                    un_y_test=un_y_test,    
                    un_y_pred=un_y_test_pred,
                    pros_results=results,
                    pros_test=y_test,
                    pros_pred=y_test_pred,
                    save_path="app/static/matrix.jpeg",
                    model_name=model_display_name
                )
            else:  # classification
                visualize_classification_results(
                    un_results=un_results,
                    un_y_test=un_y_test,    
                    un_y_pred=un_y_test_pred,
                    pros_results=results,
                    pros_test=y_test,
                    pros_pred=y_test_pred,
                    save_path="app/static/matrix.jpeg",
                    model_name=model_display_name
                )
            
            print("Visualization generated successfully")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Warning: Visualization failed but returning results anyway: {str(e)}")
            # Don't fail the entire request if visualization fails
            results["visualization_error"] = f"Visualization generation failed: {str(e)}"

        return jsonify(results)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": f"Unexpected error: {str(e)}"
        }), 500

#_______________________________________________________________________________________________________________________________________--
 
@app.route("/fetch-dataset", methods=["POST"])
def fetch_dataset():
    data = request.json
    print("Received data:", data)  

    dataset_url = data.get("url")
    if not dataset_url:
        return jsonify({"success": False, "error": "No URL provided"}), 400

    try:
        response = requests.get(dataset_url, stream=True)
        response.raise_for_status()  # Raises error for HTTP failures

        filename = "user_data.csv"
        filepath = os.path.join(os.getcwd(),"app","uploads", filename)

        with open(filepath, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        return jsonify({"success": True, "filename": filename})

    except requests.exceptions.RequestException as e:
        print("Request failed:", str(e))  # Debugging: Print error
        return jsonify({"success": False, "error": str(e)}), 500
    
@app.route('/save-file', methods=['POST'])
def save_file():
    try:
        json_data = request.get_json()
        file_path = "output/submitted_data.json"
        
        # Write the JSON data to a file
        with open(file_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        return jsonify({
            'message': 'File saved successfully',
            'path': file_path
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500
    
@app.route('/capture', methods=['POST'])
def capture():
    data = request.json
    button_text = data.get("buttonText", "Unknown Button")
    parent_div = data.get("parentDiv", "Unknown Div")
    input_value = data.get("inputVal", "No Input")

    result_array = [button_text, parent_div, input_value]

    session["result_array"] = result_array  
    session.modified = True

    print(f"Received Data: {result_array}")

    return jsonify(result_array)

@app.route('/get_matrix_image')
def get_matrix_image():
    """Serve the matrix comparison image"""
    matrix_path = os.path.join(os.getcwd(), "app", "static", "matrix.jpeg")
    if os.path.exists(matrix_path):
        return send_file(matrix_path, mimetype='image/jpeg')
    else:
        return "Image not found", 404

@app.route('/checkbox-data', methods=['POST'])
def checkbox_data():
    data = request.get_json()
    generate_value = data.get("generate", 0) 

    global_store.global_data["checkbox"] = generate_value  
    print(f"Checkbox Value Received: {generate_value}")
    
    # If checkbox is checked, immediately process data with synthetic generation
    if generate_value == 1:
        try:
            from pre_processing.main import main
            print("Processing data with synthetic generation...")
            processed_df, _ = main("app/uploads/user_data.csv", gen_syn_data=True)
            print("Data processing with synthetic generation completed!")
            return jsonify({
                "message": "Synthetic data generated and saved to clean_user_data.csv", 
                "value": generate_value,
                "rows": len(processed_df)
            })
        except Exception as e:
            print(f"Error during synthetic data generation: {e}")
            return jsonify({
                "message": f"Error generating synthetic data: {str(e)}", 
                "value": generate_value
            }), 500
    else:
        # If checkbox is unchecked, process data without synthetic generation
        try:
            from pre_processing.main import main
            print("Processing data without synthetic generation...")
            processed_df, _ = main("app/uploads/user_data.csv", gen_syn_data=False)
            print("Data processing completed!")
            return jsonify({
                "message": "Data processed without synthetic generation", 
                "value": generate_value,
                "rows": len(processed_df)
            })
        except Exception as e:
            print(f"Error during data processing: {e}")
            return jsonify({
                "message": f"Error processing data: {str(e)}", 
                "value": generate_value
            }), 500
    

def app_run():
    app.run(debug=False, port=5000)

if __name__ == "__main__":
    app.run(debug=False, port=5004)




  

