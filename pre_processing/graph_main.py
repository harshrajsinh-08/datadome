import os
import pandas as pd
import networkx as nx
from pre_processing.modules.graph_processing import GraphDatasetCleaner
from pre_processing.modules.graph_models import GraphMLTrainer

def process_graph_dataset(graph_file_path,
                         graph_format='edgelist',
                         directed=False,
                         weighted=False,
                         remove_isolated_nodes=True,
                         remove_self_loops=True,
                         remove_duplicate_edges=True,
                         min_degree_threshold=1,
                         keep_largest_component=True,
                         min_component_size=5,
                         train_models=False,
                         ml_task_type='degree_classification',
                         max_nodes_for_ml=1000):
    """
    Main function to process and clean a graph dataset
    
    Args:
        graph_file_path: Path to the graph file
        graph_format: Format of the graph file ('edgelist', 'csv', 'gml', 'graphml')
        directed: Whether the graph is directed
        weighted: Whether the graph has edge weights
        remove_isolated_nodes: Remove nodes with no connections
        remove_self_loops: Remove self-loop edges
        remove_duplicate_edges: Remove duplicate edges
        min_degree_threshold: Minimum degree for nodes to keep
        keep_largest_component: Keep only the largest connected component
        min_component_size: Minimum size for components to keep
        train_models: Whether to train ML models for comparison
        ml_task_type: Type of ML task ('degree_classification', 'centrality_classification', 'community_detection')
        max_nodes_for_ml: Maximum number of nodes to use for ML training
    
    Returns:
        tuple: (original_analysis, cleaned_analysis, cleaning_stats, ml_comparison_results)
    """
    
    print(f"Starting DataDome Graph Dataset Processing...")
    print(f"Graph file: {graph_file_path}")
    
    # Initialize the graph cleaner
    cleaner = GraphDatasetCleaner()
    
    # Step 1: Load the graph
    print("\n=== Step 1: Loading Graph ===")
    
    if graph_format == 'gml':
        success = cleaner.load_graph_from_gml(graph_file_path)
    else:
        success = cleaner.load_graph_from_edgelist(graph_file_path, directed, weighted)
    
    if not success:
        print("❌ Failed to load graph")
        return None, None, None, None
    
    # Step 2: Analyze original graph quality
    print("\n=== Step 2: Analyzing Original Graph ===")
    original_analysis = cleaner.analyze_graph_quality()
    
    if original_analysis:
        print("Original Graph Analysis:")
        print(f"  Nodes: {original_analysis['basic_stats']['nodes']}")
        print(f"  Edges: {original_analysis['basic_stats']['edges']}")
        print(f"  Density: {original_analysis['basic_stats']['density']:.4f}")
        print(f"  Connected: {original_analysis['connectivity']['is_connected']}")
        print(f"  Components: {original_analysis['connectivity']['num_components']}")
        print(f"  Isolated nodes: {original_analysis['quality_issues']['isolated_nodes']}")
        print(f"  Self loops: {original_analysis['quality_issues']['self_loops']}")
        print(f"  Average degree: {original_analysis['degree_stats']['avg_degree']:.2f}")
    
    # Step 3: Clean the graph
    print("\n=== Step 3: Cleaning Graph ===")
    cleaning_success = cleaner.clean_graph(
        remove_isolated_nodes=remove_isolated_nodes,
        remove_self_loops=remove_self_loops,
        remove_duplicate_edges=remove_duplicate_edges,
        min_degree_threshold=min_degree_threshold,
        keep_largest_component=keep_largest_component,
        min_component_size=min_component_size
    )
    
    if not cleaning_success:
        print("❌ Graph cleaning failed")
        return original_analysis, None, None, None
    
    # Step 4: Analyze cleaned graph
    print("\n=== Step 4: Analyzing Cleaned Graph ===")
    cleaned_analysis = cleaner.analyze_graph_quality()
    
    if cleaned_analysis:
        print("Cleaned Graph Analysis:")
        print(f"  Nodes: {cleaned_analysis['basic_stats']['nodes']}")
        print(f"  Edges: {cleaned_analysis['basic_stats']['edges']}")
        print(f"  Density: {cleaned_analysis['basic_stats']['density']:.4f}")
        print(f"  Connected: {cleaned_analysis['connectivity']['is_connected']}")
        print(f"  Components: {cleaned_analysis['connectivity']['num_components']}")
        print(f"  Average degree: {cleaned_analysis['degree_stats']['avg_degree']:.2f}")
    
    # Step 5: Export cleaned graph
    print("\n=== Step 5: Exporting Cleaned Graph ===")
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Export in multiple formats
    cleaner.export_cleaned_graph("output/cleaned_graph.csv", format='csv')
    cleaner.export_cleaned_graph("output/cleaned_graph.edgelist", format='edgelist')
    
    try:
        cleaner.export_cleaned_graph("output/cleaned_graph.gml", format='gml')
    except:
        print("⚠️ Could not export as GML (node IDs might not be strings)")
    
    # Step 6: ML Model Training and Comparison (Optional)
    ml_comparison_results = None
    
    if train_models:
        print("\n=== Step 6: ML Model Training & Comparison ===")
        
        trainer = GraphMLTrainer()
        
        ml_comparison_results = trainer.compare_graph_datasets(
            cleaner.original_graph,
            cleaner.graph,
            task_type=ml_task_type,
            max_nodes=max_nodes_for_ml
        )
        
        if ml_comparison_results:
            # Create visualization
            viz_dir = "app/static/graph_results"
            os.makedirs(viz_dir, exist_ok=True)
            
            viz_path = os.path.join(viz_dir, "graph_ml_comparison.png")
            trainer.visualize_graph_comparison(ml_comparison_results, viz_path)
            
            # Save results
            if ml_comparison_results['comparison']:
                comparison_df = pd.DataFrame(ml_comparison_results['comparison']).T
                comparison_df.to_csv("output/graph_ml_comparison.csv")
                print("Graph ML comparison saved to: output/graph_ml_comparison.csv")
    
    # Step 7: Save analysis results
    print("\n=== Step 7: Saving Results ===")
    
    # Save original analysis
    if original_analysis:
        original_df = pd.DataFrame([original_analysis['basic_stats']])
        original_df.to_csv("output/original_graph_analysis.csv", index=False)
        print("Original graph analysis saved to: output/original_graph_analysis.csv")
    
    # Save cleaned analysis
    if cleaned_analysis:
        cleaned_df = pd.DataFrame([cleaned_analysis['basic_stats']])
        cleaned_df.to_csv("output/cleaned_graph_analysis.csv", index=False)
        print("Cleaned graph analysis saved to: output/cleaned_graph_analysis.csv")
    
    # Save cleaning statistics
    cleaning_stats_df = pd.DataFrame([cleaner.cleaning_stats])
    cleaning_stats_df.to_csv("output/graph_cleaning_stats.csv", index=False)
    print("Cleaning statistics saved to: output/graph_cleaning_stats.csv")
    
    print(f"\n=== DataDome Graph Processing Complete ===")
    print(f"Cleaned graph saved to: output/")
    if train_models and ml_comparison_results:
        print(f"ML comparison results saved to: output/graph_ml_comparison.csv")
        print(f"Visualizations saved to: app/static/graph_results/")
    
    return original_analysis, cleaned_analysis, cleaner.cleaning_stats, ml_comparison_results

def get_graph_processing_summary(cleaning_stats, original_analysis, cleaned_analysis, ml_comparison_results=None):
    """Generate a summary of the graph processing results"""
    
    if not cleaning_stats:
        return "No processing results available"
    
    original_nodes = cleaning_stats['original_nodes']
    final_nodes = cleaning_stats['final_nodes']
    original_edges = cleaning_stats['original_edges']
    final_edges = cleaning_stats['final_edges']
    
    nodes_removed = original_nodes - final_nodes
    edges_removed = original_edges - final_edges
    
    node_removal_pct = (nodes_removed / original_nodes) * 100 if original_nodes > 0 else 0
    edge_removal_pct = (edges_removed / original_edges) * 100 if original_edges > 0 else 0
    
    summary = f"""
DataDome Graph Dataset Processing Summary:

📊 Graph Overview:
   • Original nodes: {original_nodes:,}
   • Final nodes: {final_nodes:,}
   • Nodes removed: {nodes_removed:,} ({node_removal_pct:.1f}%)
   • Original edges: {original_edges:,}
   • Final edges: {final_edges:,}
   • Edges removed: {edges_removed:,} ({edge_removal_pct:.1f}%)

🧹 Cleaning Breakdown:
   • Isolated nodes removed: {cleaning_stats['isolated_nodes_removed']}
   • Self loops removed: {cleaning_stats['self_loops_removed']}
   • Duplicate edges removed: {cleaning_stats['duplicate_edges_removed']}
   • Low degree nodes removed: {cleaning_stats['low_degree_nodes_removed']}
   • Small components removed: {cleaning_stats['disconnected_components_removed']}

📈 Graph Quality Metrics:"""

    if original_analysis and cleaned_analysis:
        orig_density = original_analysis['basic_stats']['density']
        clean_density = cleaned_analysis['basic_stats']['density']
        orig_components = original_analysis['connectivity']['num_components']
        clean_components = cleaned_analysis['connectivity']['num_components']
        
        summary += f"""
   • Original density: {orig_density:.4f}
   • Cleaned density: {clean_density:.4f}
   • Original components: {orig_components}
   • Cleaned components: {clean_components}
   • Connectivity improved: {'Yes' if clean_components < orig_components else 'No'}"""

    # Add ML model comparison results if available
    if ml_comparison_results and ml_comparison_results.get('comparison'):
        comparison = ml_comparison_results['comparison']
        avg_improvement = sum(comp['improvement_percentage'] for comp in comparison.values()) / len(comparison)
        best_model = max(comparison.keys(), key=lambda k: comparison[k]['improvement_percentage'])
        best_improvement = comparison[best_model]['improvement_percentage']
        task_type = ml_comparison_results.get('task_type', 'classification')
        
        summary += f"""

🤖 ML Model Performance Comparison:
   • Task type: {task_type.replace('_', ' ').title()}
   • Models trained: {len(comparison)}
   • Average accuracy improvement: {avg_improvement:.1f}%
   • Best performing model: {best_model}
   • Best improvement: {best_improvement:.1f}%
   • Training samples: {ml_comparison_results['original_samples']} → {ml_comparison_results['cleaned_samples']}"""

    summary += f"""

✅ Results:
   • Cleaned graph saved to: output/cleaned_graph.csv
   • Analysis reports saved to: output/
   • Visualizations saved to: app/static/graph_results/
"""
    
    return summary