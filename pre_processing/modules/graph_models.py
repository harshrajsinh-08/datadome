import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class GraphMLTrainer:
    def __init__(self):
        self.classification_models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='rbf', random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        
        self.regression_models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf'),
            'Linear Regression': LinearRegression(),
            'KNN': KNeighborsRegressor(n_neighbors=5)
        }
    
    def extract_node_features(self, graph, max_nodes=1000):
        """Extract features for each node in the graph"""
        if graph.number_of_nodes() == 0:
            return None, None, None
        
        print(f"Extracting features from {graph.number_of_nodes()} nodes...")
        
        # Limit nodes if graph is too large
        nodes = list(graph.nodes())
        if len(nodes) > max_nodes:
            nodes = nodes[:max_nodes]
            subgraph = graph.subgraph(nodes)
        else:
            subgraph = graph
        
        features = []
        node_ids = []
        
        # Calculate centrality measures
        print("Calculating centrality measures...")
        try:
            degree_centrality = nx.degree_centrality(subgraph)
            betweenness_centrality = nx.betweenness_centrality(subgraph)
            closeness_centrality = nx.closeness_centrality(subgraph)
            eigenvector_centrality = nx.eigenvector_centrality(subgraph, max_iter=1000)
        except:
            # Fallback for problematic graphs
            degree_centrality = {node: subgraph.degree(node) / (subgraph.number_of_nodes() - 1) for node in subgraph.nodes()}
            betweenness_centrality = {node: 0 for node in subgraph.nodes()}
            closeness_centrality = {node: 0 for node in subgraph.nodes()}
            eigenvector_centrality = {node: 0 for node in subgraph.nodes()}
        
        # Calculate clustering coefficient
        if not subgraph.is_directed():
            clustering = nx.clustering(subgraph)
        else:
            clustering = {node: 0 for node in subgraph.nodes()}
        
        # Calculate PageRank
        try:
            pagerank = nx.pagerank(subgraph)
        except:
            pagerank = {node: 1/subgraph.number_of_nodes() for node in subgraph.nodes()}
        
        # Extract features for each node
        for node in subgraph.nodes():
            # Basic features
            node_features = [
                subgraph.degree(node),  # Total degree
                degree_centrality.get(node, 0),  # Degree centrality
                betweenness_centrality.get(node, 0),  # Betweenness centrality
                closeness_centrality.get(node, 0),  # Closeness centrality
                eigenvector_centrality.get(node, 0),  # Eigenvector centrality
                clustering.get(node, 0),  # Clustering coefficient
                pagerank.get(node, 0),  # PageRank
            ]
            
            # Add directed graph specific features
            if subgraph.is_directed():
                in_deg = subgraph.in_degree(node)
                out_deg = subgraph.out_degree(node)
                node_features.extend([
                    in_deg,  # In-degree
                    out_deg,  # Out-degree
                    in_deg / (in_deg + out_deg) if (in_deg + out_deg) > 0 else 0.5,  # In-degree ratio
                ])
            
            # Add local neighborhood features
            neighbors = list(subgraph.neighbors(node))
            node_features.extend([
                len(neighbors),  # Number of neighbors
                np.mean([subgraph.degree(n) for n in neighbors]) if neighbors else 0,  # Avg neighbor degree
                len(set(neighbors) & set(subgraph.neighbors(node))) if neighbors else 0,  # Common neighbors
            ])
            
            features.append(node_features)
            node_ids.append(node)
        
        # Build feature names list
        feature_names = [
            'degree', 'degree_centrality', 'betweenness_centrality', 'closeness_centrality',
            'eigenvector_centrality', 'clustering_coefficient', 'pagerank'
        ]
        
        # Add directed graph feature names
        if subgraph.is_directed():
            feature_names.extend(['in_degree', 'out_degree', 'in_degree_ratio'])
        
        # Add neighborhood feature names
        feature_names.extend(['num_neighbors', 'avg_neighbor_degree', 'common_neighbors'])
        
        return np.array(features), node_ids, feature_names
    
    def generate_node_labels(self, graph, node_ids, task_type='degree_classification'):
        """Generate labels for nodes based on graph structure"""
        labels = []
        
        if task_type == 'degree_classification':
            # For directed graphs, use a more nuanced classification based on in/out degree
            if graph.is_directed():
                # Classify based on node role in directed graph
                for node in node_ids:
                    in_deg = graph.in_degree(node)
                    out_deg = graph.out_degree(node)
                    total_deg = in_deg + out_deg
                    
                    # Classify by role
                    if in_deg == 0 and out_deg > 0:
                        labels.append('Source')  # Only outgoing edges
                    elif out_deg == 0 and in_deg > 0:
                        labels.append('Sink')  # Only incoming edges
                    elif in_deg > out_deg * 2:
                        labels.append('Receiver')  # Mostly incoming
                    elif out_deg > in_deg * 2:
                        labels.append('Sender')  # Mostly outgoing
                    else:
                        labels.append('Balanced')  # Balanced in/out
            else:
                # For undirected graphs, use degree-based classification
                degrees = [graph.degree(node) for node in node_ids]
                degree_percentiles = np.percentile(degrees, [33, 67])
                
                for node in node_ids:
                    degree = graph.degree(node)
                    if degree <= degree_percentiles[0]:
                        labels.append('Low_Degree')
                    elif degree <= degree_percentiles[1]:
                        labels.append('Medium_Degree')
                    else:
                        labels.append('High_Degree')
        
        elif task_type == 'centrality_classification':
            # Classify nodes based on betweenness centrality
            try:
                betweenness = nx.betweenness_centrality(graph)
                centrality_values = [betweenness.get(node, 0) for node in node_ids]
                centrality_percentiles = np.percentile(centrality_values, [50])
                
                for node in node_ids:
                    centrality = betweenness.get(node, 0)
                    if centrality <= centrality_percentiles[0]:
                        labels.append('Peripheral')
                    else:
                        labels.append('Central')
            except:
                # Fallback to degree-based classification
                return self.generate_node_labels(graph, node_ids, 'degree_classification')
        
        elif task_type == 'community_detection':
            # Use community detection algorithms
            try:
                if not graph.is_directed():
                    communities = nx.community.greedy_modularity_communities(graph)
                    node_to_community = {}
                    for i, community in enumerate(communities):
                        for node in community:
                            node_to_community[node] = f'Community_{i}'
                    
                    labels = [node_to_community.get(node, 'Unknown') for node in node_ids]
                else:
                    # Fallback for directed graphs
                    return self.generate_node_labels(graph, node_ids, 'degree_classification')
            except:
                return self.generate_node_labels(graph, node_ids, 'degree_classification')
        
        return labels
    
    def train_node_classification(self, features, labels):
        """Train classification models on node features"""
        print(f"Training classification models on {len(features)} nodes...")
        
        # Check if we have enough samples
        unique_labels, counts = np.unique(labels, return_counts=True)
        min_samples = min(counts)
        
        if len(features) < 20:
            print("⚠️ Too few samples for reliable training")
            return None
        
        if min_samples < 2:
            print("⚠️ Some classes have only 1 sample. Filtering...")
            # Filter out singleton classes
            valid_indices = []
            for i, label in enumerate(labels):
                label_count = counts[unique_labels == label][0]
                if label_count >= 2:
                    valid_indices.append(i)
            
            if len(valid_indices) < 10:
                print("❌ Too few samples after filtering")
                return None
            
            features = features[valid_indices]
            labels = np.array(labels)[valid_indices]
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )
        except:
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42
            )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        
        for model_name, model in self.classification_models.items():
            try:
                print(f"Training {model_name}...")
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'classification_report': report,
                    'y_test': y_test,
                    'y_pred': y_pred
                }
                
                print(f"{model_name} - Accuracy: {accuracy:.3f}")
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                results[model_name] = None
        
        return results
    
    def compare_graph_datasets(self, original_graph, cleaned_graph, task_type='degree_classification', max_nodes=1000):
        """Compare ML performance on original vs cleaned graphs"""
        print("=== Comparing Graph Dataset Performance ===")
        
        # Extract features from original graph
        print("\n1. Processing Original Graph...")
        orig_features, orig_node_ids, feature_names = self.extract_node_features(original_graph, max_nodes)
        
        if orig_features is None:
            print("❌ No features extracted from original graph")
            return None
        
        orig_labels = self.generate_node_labels(original_graph, orig_node_ids, task_type)
        
        # Extract features from cleaned graph
        print("\n2. Processing Cleaned Graph...")
        clean_features, clean_node_ids, _ = self.extract_node_features(cleaned_graph, max_nodes)
        
        if clean_features is None:
            print("❌ No features extracted from cleaned graph")
            return None
        
        clean_labels = self.generate_node_labels(cleaned_graph, clean_node_ids, task_type)
        
        # Train models on original graph
        print("\n3. Training Models on Original Graph...")
        original_results = self.train_node_classification(orig_features, orig_labels)
        
        # Train models on cleaned graph
        print("\n4. Training Models on Cleaned Graph...")
        cleaned_results = self.train_node_classification(clean_features, clean_labels)
        
        if original_results is None or cleaned_results is None:
            print("❌ Model training failed")
            return None
        
        # Compare results
        print("\n5. Comparing Results...")
        comparison = self._compare_results(original_results, cleaned_results)
        
        return {
            'original_results': original_results,
            'cleaned_results': cleaned_results,
            'comparison': comparison,
            'original_samples': len(orig_features),
            'cleaned_samples': len(clean_features),
            'task_type': task_type,
            'feature_names': feature_names
        }
    
    def _compare_results(self, original_results, cleaned_results):
        """Compare results between original and cleaned graphs"""
        comparison = {}
        
        for model_name in self.classification_models.keys():
            if (model_name in original_results and original_results[model_name] and
                model_name in cleaned_results and cleaned_results[model_name]):
                
                orig_acc = original_results[model_name]['accuracy']
                clean_acc = cleaned_results[model_name]['accuracy']
                improvement = clean_acc - orig_acc
                improvement_pct = (improvement / orig_acc) * 100 if orig_acc > 0 else 0
                
                comparison[model_name] = {
                    'original_accuracy': orig_acc,
                    'cleaned_accuracy': clean_acc,
                    'improvement': improvement,
                    'improvement_percentage': improvement_pct
                }
        
        return comparison
    
    def visualize_graph_comparison(self, comparison_results, save_path):
        """Create visualization comparing graph ML performance"""
        if not comparison_results or 'comparison' not in comparison_results:
            return None
        
        comparison = comparison_results['comparison']
        
        # Create larger figure with better spacing
        fig, axes = plt.subplots(2, 2, figsize=(18, 15))
        axes = axes.flatten()
        
        # Main title with better formatting
        task_type = comparison_results.get('task_type', 'Node Classification')
        fig.suptitle(f'🕸️ DataDome Graph Dataset: {task_type.replace("_", " ").title()} Performance', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Accuracy Comparison
        models = list(comparison.keys())
        original_accs = [comparison[model]['original_accuracy'] for model in models]
        cleaned_accs = [comparison[model]['cleaned_accuracy'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = axes[0].bar(x - width/2, original_accs, width, label='Original Graph', 
                           color='#FF6B6B', alpha=0.8, edgecolor='darkred', linewidth=1.5)
        bars2 = axes[0].bar(x + width/2, cleaned_accs, width, label='Cleaned Graph', 
                           color='#4ECDC4', alpha=0.8, edgecolor='darkcyan', linewidth=1.5)
        
        axes[0].set_title('📊 Model Accuracy Comparison', fontsize=14, pad=20, fontweight='bold')
        axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Models', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        axes[0].legend(fontsize=11, framealpha=0.9)
        axes[0].grid(axis='y', alpha=0.3, linestyle='--')
        axes[0].set_ylim([0, 1.1])  # Set consistent y-axis
        
        # Add value labels with better positioning
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 2. Improvement Percentage with better color coding
        improvements = [comparison[model]['improvement_percentage'] for model in models]
        colors = ['#27ae60' if imp > 0.1 else '#e74c3c' if imp < -0.1 else '#95a5a6' for imp in improvements]
        
        bars = axes[1].bar(models, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[1].set_title('📈 Performance Improvement (%)', fontsize=14, pad=20, fontweight='bold')
        axes[1].set_ylabel('Improvement Percentage', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Models', fontsize=12, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45, labelsize=10)
        axes[1].grid(axis='y', alpha=0.3, linestyle='--')
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.5)
        
        # Add value labels with better positioning
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            label_y = height + (1 if height > 0 else -2)
            axes[1].text(bar.get_x() + bar.get_width()/2., label_y,
                        f'{imp:.2f}%', ha='center', 
                        va='bottom' if height > 0 else 'top', fontsize=9, fontweight='bold')
        
        # 3. Dataset Size Comparison with better visualization
        dataset_sizes = [
            comparison_results['original_samples'],
            comparison_results['cleaned_samples']
        ]
        dataset_labels = ['Original\nGraph', 'Cleaned\nGraph']
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = axes[2].bar(dataset_labels, dataset_sizes, color=colors, alpha=0.8, 
                          edgecolor='black', linewidth=2, width=0.6)
        axes[2].set_title('📦 Dataset Size Comparison', fontsize=14, pad=20, fontweight='bold')
        axes[2].set_ylabel('Number of Nodes', fontsize=12, fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels with better formatting
        for bar, size in zip(bars, dataset_sizes):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., height + max(dataset_sizes)*0.02,
                        f'{size:,}', ha='center', va='bottom', fontsize=13, fontweight='bold')
        
        # Add reduction percentage
        if dataset_sizes[0] > 0:
            reduction_pct = ((dataset_sizes[0] - dataset_sizes[1]) / dataset_sizes[0]) * 100
            axes[2].text(0.5, 0.95, f'Reduction: {reduction_pct:.1f}%', 
                        transform=axes[2].transAxes, ha='center', va='top',
                        fontsize=11, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # 4. Enhanced Performance Summary
        avg_improvement = np.mean(improvements)
        best_model = models[np.argmax(improvements)]
        best_improvement = max(improvements)
        worst_model = models[np.argmin(improvements)]
        worst_improvement = min(improvements)
        
        # Clear axis and create text summary
        axes[3].axis('off')
        
        # Title
        axes[3].text(0.5, 0.95, '📋 Performance Summary', fontsize=16, fontweight='bold', 
                    transform=axes[3].transAxes, ha='center', va='top')
        
        # Summary statistics with better formatting
        summary_lines = [
            f"{'='*50}",
            f"",
            f"📊 Overall Performance:",
            f"   Average Improvement: {avg_improvement:.2f}%",
            f"   Models Evaluated: {len(models)}",
            f"",
            f"🏆 Best Performing Model:",
            f"   {best_model}",
            f"   Improvement: {best_improvement:.2f}%",
            f"",
            f"📉 Worst Performing Model:",
            f"   {worst_model}",
            f"   Change: {worst_improvement:.2f}%",
            f"",
            f"📦 Dataset Statistics:",
            f"   Original: {comparison_results['original_samples']:,} nodes",
            f"   Cleaned: {comparison_results['cleaned_samples']:,} nodes",
            f"   Removed: {comparison_results['original_samples'] - comparison_results['cleaned_samples']:,} nodes",
            f"",
            f"🎯 Task Type:",
            f"   {task_type.replace('_', ' ').title()}",
            f"",
            f"{'='*50}"
        ]
        
        summary_text = '\n'.join(summary_lines)
        
        axes[3].text(0.05, 0.85, summary_text, fontsize=10, transform=axes[3].transAxes,
                    verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle="round,pad=0.8", facecolor='#f0f0f0', 
                             edgecolor='#333333', linewidth=2, alpha=0.9))
        
        # Adjust layout for better spacing
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.35, wspace=0.3)
        
        # Save with high quality
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        
        print(f"✅ Graph ML comparison visualization saved to: {save_path}")
        return save_path