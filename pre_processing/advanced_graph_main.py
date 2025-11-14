import os
import pandas as pd
import networkx as nx
from pre_processing.modules.neo4j_graph_processing import Neo4jGraphProcessor
from pre_processing.modules.advanced_graph_models import AdvancedGraphMLTrainer
import json
from typing import Dict, Optional, Any

def process_advanced_graph_dataset(
    graph_file_path: str,
    # Neo4j connection parameters
    neo4j_uri: Optional[str] = None,
    neo4j_user: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    # Graph loading parameters
    directed: bool = False,
    weighted: bool = False,
    # Advanced cleaning parameters
    remove_isolated_nodes: bool = True,
    remove_self_loops: bool = True,
    remove_duplicate_edges: bool = True,
    min_degree_threshold: int = 1,
    max_degree_threshold: Optional[int] = None,
    keep_largest_component: bool = True,
    min_component_size: int = 5,
    remove_bridges: bool = False,
    anomaly_detection: bool = True,
    preserve_important_nodes: bool = True,
    # ML parameters
    train_models: bool = True,
    ml_task_type: str = 'centrality_classification',
    max_nodes_for_ml: int = 2000,
    use_neo4j_analytics: bool = False,
    export_to_neo4j: bool = False
) -> Dict[str, Any]:
    """
    Advanced graph processing pipeline with Neo4j integration and modern ML techniques
    
    Args:
        graph_file_path: Path to the graph file
        neo4j_uri: Neo4j database URI (e.g., "bolt://localhost:7687")
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        directed: Whether the graph is directed
        weighted: Whether the graph has edge weights
        remove_isolated_nodes: Remove nodes with no connections
        remove_self_loops: Remove self-loop edges
        remove_duplicate_edges: Remove duplicate edges
        min_degree_threshold: Minimum degree for nodes to keep
        max_degree_threshold: Maximum degree for nodes to keep (None = no limit)
        keep_largest_component: Keep only the largest connected component
        min_component_size: Minimum size for components to keep
        remove_bridges: Whether to remove bridge edges
        anomaly_detection: Use statistical anomaly detection
        preserve_important_nodes: Preserve structurally important nodes
        train_models: Whether to train ML models for comparison
        ml_task_type: Type of ML task ('centrality_classification', 'community_role', 'structural_role')
        max_nodes_for_ml: Maximum number of nodes to use for ML training
        use_neo4j_analytics: Use Neo4j's built-in graph algorithms
        export_to_neo4j: Export results to Neo4j database
    
    Returns:
        Dictionary containing all processing results
    """
    
    print(f"🚀 Starting DataDome Advanced Graph Processing...")
    print(f"Graph file: {graph_file_path}")
    print(f"Neo4j integration: {'Enabled' if neo4j_uri else 'Disabled'}")
    
    # Initialize the advanced graph processor
    processor = Neo4jGraphProcessor(neo4j_uri, neo4j_user, neo4j_password)
    
    # Step 1: Load the graph
    print("\n=== Step 1: Advanced Graph Loading ===")
    
    success = processor.load_graph_advanced(
        graph_file_path, 
        directed=directed, 
        weighted=weighted
    )
    
    if not success:
        print("❌ Failed to load graph")
        return None
    
    # Step 2: Analyze original graph with advanced metrics
    print("\n=== Step 2: Advanced Graph Analysis ===")
    original_metrics = processor.calculate_advanced_metrics()
    
    if original_metrics:
        print("Original Graph Advanced Metrics:")
        print(f"  Nodes: {original_metrics.get('nodes', 0):,}")
        print(f"  Edges: {original_metrics.get('edges', 0):,}")
        print(f"  Density: {original_metrics.get('density', 0):.6f}")
        print(f"  Average degree: {original_metrics.get('degree_mean', 0):.2f}")
        print(f"  Degree Gini coefficient: {original_metrics.get('degree_gini', 0):.3f}")
        
        if 'is_connected' in original_metrics:
            print(f"  Connected: {original_metrics['is_connected']}")
        if 'num_connected_components' in original_metrics:
            print(f"  Components: {original_metrics['num_connected_components']}")
        if 'average_clustering' in original_metrics:
            print(f"  Average clustering: {original_metrics['average_clustering']:.3f}")
        if 'small_world_coefficient' in original_metrics:
            print(f"  Small-world coefficient: {original_metrics['small_world_coefficient']:.3f}")
    
    # Step 3: Advanced graph cleaning
    print("\n=== Step 3: Advanced Graph Cleaning ===")
    cleaning_success = processor.advanced_graph_cleaning(
        remove_isolated_nodes=remove_isolated_nodes,
        remove_self_loops=remove_self_loops,
        remove_duplicate_edges=remove_duplicate_edges,
        min_degree_threshold=min_degree_threshold,
        max_degree_threshold=max_degree_threshold,
        keep_largest_component=keep_largest_component,
        min_component_size=min_component_size,
        remove_bridges=remove_bridges,
        anomaly_detection=anomaly_detection,
        preserve_important_nodes=preserve_important_nodes
    )
    
    if not cleaning_success:
        print("❌ Advanced graph cleaning failed")
        return None
    
    # Step 4: Analyze cleaned graph
    print("\n=== Step 4: Cleaned Graph Analysis ===")
    cleaned_metrics = processor.calculate_advanced_metrics()
    
    if cleaned_metrics:
        print("Cleaned Graph Advanced Metrics:")
        print(f"  Nodes: {cleaned_metrics.get('nodes', 0):,}")
        print(f"  Edges: {cleaned_metrics.get('edges', 0):,}")
        print(f"  Density: {cleaned_metrics.get('density', 0):.6f}")
        print(f"  Average degree: {cleaned_metrics.get('degree_mean', 0):.2f}")
        print(f"  Degree Gini coefficient: {cleaned_metrics.get('degree_gini', 0):.3f}")
        
        if 'is_connected' in cleaned_metrics:
            print(f"  Connected: {cleaned_metrics['is_connected']}")
        if 'num_connected_components' in cleaned_metrics:
            print(f"  Components: {cleaned_metrics['num_connected_components']}")
        if 'average_clustering' in cleaned_metrics:
            print(f"  Average clustering: {cleaned_metrics['average_clustering']:.3f}")
        if 'small_world_coefficient' in cleaned_metrics:
            print(f"  Small-world coefficient: {cleaned_metrics['small_world_coefficient']:.3f}")
    
    # Step 5: Export cleaned graph in multiple formats
    print("\n=== Step 5: Exporting Advanced Formats ===")
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Export in multiple advanced formats
    exported_files = processor.export_advanced_formats("output")
    
    # Step 6: Neo4j Integration
    neo4j_analytics = {}
    if neo4j_uri and export_to_neo4j:
        print("\n=== Step 6: Neo4j Integration ===")
        
        # Export to Neo4j
        export_success = processor.export_to_neo4j(clear_existing=True)
        
        if export_success and use_neo4j_analytics:
            print("Running Neo4j advanced analytics...")
            neo4j_analytics = processor.run_neo4j_analytics()
            
            if neo4j_analytics:
                print("Neo4j Analytics Results:")
                for metric, value in neo4j_analytics.items():
                    print(f"  {metric}: {value}")
    
    # Step 7: Advanced ML Model Training and Comparison
    ml_comparison_results = None
    
    if train_models:
        print(f"\n=== Step 7: Advanced ML Training & Comparison ===")
        print(f"Task type: {ml_task_type}")
        
        trainer = AdvancedGraphMLTrainer()
        
        ml_comparison_results = trainer.compare_advanced_datasets(
            processor.original_graph,
            processor.graph,
            task_type=ml_task_type,
            max_nodes=max_nodes_for_ml
        )
        
        if ml_comparison_results:
            # Create advanced visualization
            viz_dir = "app/static/graph_results"
            os.makedirs(viz_dir, exist_ok=True)
            
            viz_path = os.path.join(viz_dir, "advanced_graph_ml_comparison.png")
            trainer.create_advanced_visualization(ml_comparison_results, viz_path)
            
            # Save detailed results
            results_path = "output/advanced_graph_ml_results.json"
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in ml_comparison_results.items():
                if key in ['original_results', 'cleaned_results']:
                    serializable_results[key] = {}
                    for model_name, model_result in value.items():
                        if model_result:
                            serializable_model = {}
                            for metric, metric_value in model_result.items():
                                if metric in ['y_test', 'y_pred']:
                                    serializable_model[metric] = list(metric_value) if hasattr(metric_value, '__iter__') else metric_value
                                elif metric == 'confusion_matrix':
                                    serializable_model[metric] = metric_value  # Already converted to list
                                else:
                                    serializable_model[metric] = metric_value
                            serializable_results[key][model_name] = serializable_model
                        else:
                            serializable_results[key][model_name] = None
                else:
                    serializable_results[key] = value
            
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            print(f"Advanced ML results saved to: {results_path}")
            
            # Save comparison CSV
            if ml_comparison_results['comparison']:
                comparison_df = pd.DataFrame(ml_comparison_results['comparison']).T
                comparison_df.to_csv("output/advanced_graph_ml_comparison.csv")
                print("Advanced ML comparison saved to: output/advanced_graph_ml_comparison.csv")
    
    # Step 8: Save comprehensive analysis results
    print("\n=== Step 8: Saving Comprehensive Results ===")
    
    # Save original metrics
    if original_metrics:
        original_df = pd.DataFrame([original_metrics])
        original_df.to_csv("output/original_graph_advanced_metrics.csv", index=False)
        print("Original graph metrics saved to: output/original_graph_advanced_metrics.csv")
    
    # Save cleaned metrics
    if cleaned_metrics:
        cleaned_df = pd.DataFrame([cleaned_metrics])
        cleaned_df.to_csv("output/cleaned_graph_advanced_metrics.csv", index=False)
        print("Cleaned graph metrics saved to: output/cleaned_graph_advanced_metrics.csv")
    
    # Save cleaning statistics
    cleaning_stats_df = pd.DataFrame([processor.cleaning_stats])
    cleaning_stats_df.to_csv("output/advanced_graph_cleaning_stats.csv", index=False)
    print("Advanced cleaning statistics saved to: output/advanced_graph_cleaning_stats.csv")
    
    # Save Neo4j analytics if available
    if neo4j_analytics:
        neo4j_df = pd.DataFrame([neo4j_analytics])
        neo4j_df.to_csv("output/neo4j_analytics_results.csv", index=False)
        print("Neo4j analytics saved to: output/neo4j_analytics_results.csv")
    
    # Compile comprehensive results
    comprehensive_results = {
        'original_metrics': original_metrics,
        'cleaned_metrics': cleaned_metrics,
        'cleaning_stats': processor.cleaning_stats,
        'neo4j_analytics': neo4j_analytics,
        'ml_comparison_results': ml_comparison_results,
        'exported_files': exported_files,
        'processing_parameters': {
            'directed': directed,
            'weighted': weighted,
            'remove_isolated_nodes': remove_isolated_nodes,
            'remove_self_loops': remove_self_loops,
            'remove_duplicate_edges': remove_duplicate_edges,
            'min_degree_threshold': min_degree_threshold,
            'max_degree_threshold': max_degree_threshold,
            'keep_largest_component': keep_largest_component,
            'min_component_size': min_component_size,
            'remove_bridges': remove_bridges,
            'anomaly_detection': anomaly_detection,
            'preserve_important_nodes': preserve_important_nodes,
            'ml_task_type': ml_task_type,
            'max_nodes_for_ml': max_nodes_for_ml
        }
    }
    
    print(f"\n=== DataDome Advanced Graph Processing Complete ===")
    print(f"Results saved to: output/")
    if train_models and ml_comparison_results:
        print(f"Advanced ML visualizations saved to: app/static/graph_results/")
    if neo4j_uri and export_to_neo4j:
        print(f"Graph exported to Neo4j database")
    
    return comprehensive_results

def get_advanced_processing_summary(results: Dict[str, Any]) -> str:
    """Generate a comprehensive summary of advanced graph processing results"""
    
    if not results:
        return "No processing results available"
    
    cleaning_stats = results.get('cleaning_stats', {})
    original_metrics = results.get('original_metrics', {})
    cleaned_metrics = results.get('cleaned_metrics', {})
    ml_results = results.get('ml_comparison_results', {})
    neo4j_analytics = results.get('neo4j_analytics', {})
    
    original_nodes = cleaning_stats.get('original_nodes', 0)
    final_nodes = cleaning_stats.get('final_nodes', 0)
    original_edges = cleaning_stats.get('original_edges', 0)
    final_edges = cleaning_stats.get('final_edges', 0)
    
    nodes_removed = original_nodes - final_nodes
    edges_removed = original_edges - final_edges
    
    node_removal_pct = (nodes_removed / original_nodes) * 100 if original_nodes > 0 else 0
    edge_removal_pct = (edges_removed / original_edges) * 100 if original_edges > 0 else 0
    
    summary = f"""
🚀 DataDome Advanced Graph Processing Summary:

📊 Graph Transformation:
   • Original nodes: {original_nodes:,}
   • Final nodes: {final_nodes:,}
   • Nodes removed: {nodes_removed:,} ({node_removal_pct:.1f}%)
   • Original edges: {original_edges:,}
   • Final edges: {final_edges:,}
   • Edges removed: {edges_removed:,} ({edge_removal_pct:.1f}%)

🧹 Advanced Cleaning Operations:
   • Isolated nodes removed: {cleaning_stats.get('isolated_nodes_removed', 0)}
   • Self loops removed: {cleaning_stats.get('self_loops_removed', 0)}
   • Duplicate edges removed: {cleaning_stats.get('duplicate_edges_removed', 0)}
   • Low degree nodes removed: {cleaning_stats.get('low_degree_nodes_removed', 0)}
   • Anomalous nodes removed: {cleaning_stats.get('anomalous_nodes_removed', 0)}
   • Small components removed: {cleaning_stats.get('disconnected_components_removed', 0)}
   • Bridge edges preserved: {cleaning_stats.get('bridge_edges_preserved', 0)}

📈 Advanced Graph Metrics:"""

    if original_metrics and cleaned_metrics:
        orig_density = original_metrics.get('density', 0)
        clean_density = cleaned_metrics.get('density', 0)
        orig_clustering = original_metrics.get('average_clustering', 0)
        clean_clustering = cleaned_metrics.get('average_clustering', 0)
        orig_gini = original_metrics.get('degree_gini', 0)
        clean_gini = cleaned_metrics.get('degree_gini', 0)
        
        summary += f"""
   • Original density: {orig_density:.6f} → Cleaned: {clean_density:.6f}
   • Original clustering: {orig_clustering:.3f} → Cleaned: {clean_clustering:.3f}
   • Original degree Gini: {orig_gini:.3f} → Cleaned: {clean_gini:.3f}"""
        
        if 'small_world_coefficient' in cleaned_metrics:
            summary += f"""
   • Small-world coefficient: {cleaned_metrics['small_world_coefficient']:.3f}"""
        
        if 'num_communities_louvain' in cleaned_metrics:
            summary += f"""
   • Communities detected: {cleaned_metrics['num_communities_louvain']}"""

    # Add Neo4j analytics if available
    if neo4j_analytics:
        summary += f"""

🔗 Neo4j Advanced Analytics:"""
        for metric, value in neo4j_analytics.items():
            if isinstance(value, float):
                summary += f"""
   • {metric.replace('_', ' ').title()}: {value:.4f}"""
            else:
                summary += f"""
   • {metric.replace('_', ' ').title()}: {value}"""

    # Add ML model comparison results if available
    if ml_results and ml_results.get('comparison'):
        comparison = ml_results['comparison']
        avg_improvement = sum(comp['accuracy_improvement_percentage'] for comp in comparison.values()) / len(comparison)
        best_model = max(comparison.keys(), key=lambda k: comparison[k]['accuracy_improvement_percentage'])
        best_improvement = comparison[best_model]['accuracy_improvement_percentage']
        task_type = ml_results.get('task_type', 'classification')
        
        # Count advanced models
        advanced_models = [m for m in comparison.keys() if m in ['XGBoost', 'LightGBM', 'Deep MLP']]
        traditional_models = [m for m in comparison.keys() if m not in advanced_models]
        
        summary += f"""

🤖 Advanced ML Performance Analysis:
   • Task type: {task_type.replace('_', ' ').title()}
   • Traditional models: {len(traditional_models)}
   • Advanced models: {len(advanced_models)}
   • Total features extracted: {ml_results.get('original_features', 0)} → {ml_results.get('cleaned_features', 0)}
   • Average accuracy improvement: {avg_improvement:.1f}%
   • Best performing model: {best_model}
   • Best improvement: {best_improvement:.1f}%
   • Training samples: {ml_results['original_samples']} → {ml_results['cleaned_samples']}"""
        
        # Highlight advanced model performance
        if advanced_models:
            summary += f"""
   
   Advanced Model Highlights:"""
            for model in advanced_models:
                if model in comparison:
                    improvement = comparison[model]['accuracy_improvement_percentage']
                    summary += f"""
   • {model}: {improvement:.1f}% improvement"""

    summary += f"""

✅ Advanced Processing Results:
   • Cleaned graph exported in multiple formats (GraphML, GEXF, JSON, CSV)
   • Comprehensive metrics analysis saved
   • Advanced ML comparison completed
   • Visualizations generated"""
    
    if neo4j_analytics:
        summary += f"""
   • Neo4j database integration successful"""
    
    summary += f"""
   • All results saved to: output/
   • Visualizations saved to: app/static/graph_results/
"""
    
    return summary

# Example usage and testing functions
def test_advanced_processing():
    """Test the advanced graph processing pipeline"""
    
    # Create a sample graph for testing
    print("Creating sample graph for testing...")
    
    # Generate a sample graph
    import networkx as nx
    import numpy as np
    
    # Create a scale-free network with communities
    G = nx.barabasi_albert_graph(500, 3)
    
    # Add some noise and anomalies
    # Add isolated nodes
    for i in range(500, 520):
        G.add_node(i)
    
    # Add self loops
    for i in range(10):
        G.add_edge(i, i)
    
    # Add high-degree hub
    hub_node = 600
    G.add_node(hub_node)
    for i in range(100):
        G.add_edge(hub_node, i)
    
    # Save test graph
    os.makedirs("test_data", exist_ok=True)
    nx.write_edgelist(G, "test_data/sample_graph.edgelist")
    
    print(f"Sample graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Test the advanced processing
    results = process_advanced_graph_dataset(
        graph_file_path="test_data/sample_graph.edgelist",
        # Neo4j parameters (uncomment if you have Neo4j running)
        # neo4j_uri="bolt://localhost:7687",
        # neo4j_user="neo4j",
        # neo4j_password="password",
        # export_to_neo4j=True,
        # use_neo4j_analytics=True,
        
        # Advanced cleaning parameters
        anomaly_detection=True,
        preserve_important_nodes=True,
        max_degree_threshold=50,  # Remove very high degree nodes
        
        # ML parameters
        train_models=True,
        ml_task_type='centrality_classification',
        max_nodes_for_ml=1000
    )
    
    if results:
        print("\n" + "="*80)
        print(get_advanced_processing_summary(results))
        print("="*80)
    
    return results

if __name__ == "__main__":
    # Run test
    test_results = test_advanced_processing()