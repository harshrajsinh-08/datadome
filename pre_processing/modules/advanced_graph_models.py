import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
from typing import Dict, List, Tuple, Optional, Any
import json

# Advanced ML imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Graph Neural Network imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool
    from torch_geometric.utils import from_networkx, to_networkx
    from torch_geometric.loader import DataLoader
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

# Community detection
try:
    import community as community_louvain
    COMMUNITY_AVAILABLE = True
except ImportError:
    COMMUNITY_AVAILABLE = False

# Node2Vec
try:
    from node2vec import Node2Vec
    NODE2VEC_AVAILABLE = True
except ImportError:
    NODE2VEC_AVAILABLE = False

warnings.filterwarnings('ignore')

class AdvancedGraphMLTrainer:
    """Advanced Graph ML with modern techniques including GNNs and embeddings"""
    
    def __init__(self):
        self.traditional_models = {
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Extra Trees': ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            'SVM (RBF)': SVC(kernel='rbf', random_state=42, probability=True),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=2000),
            'KNN': KNeighborsClassifier(n_neighbors=7),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
        }
        
        # Add advanced models if available
        if XGBOOST_AVAILABLE:
            self.traditional_models['XGBoost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        
        if LIGHTGBM_AVAILABLE:
            self.traditional_models['LightGBM'] = lgb.LGBMClassifier(random_state=42, verbose=-1)
        
        self.feature_extractors = {
            'centrality': self._extract_centrality_features,
            'structural': self._extract_structural_features,
            'community': self._extract_community_features,
            'spectral': self._extract_spectral_features,
            'motif': self._extract_motif_features
        }
        
        if NODE2VEC_AVAILABLE:
            self.feature_extractors['node2vec'] = self._extract_node2vec_features
    
    def extract_comprehensive_features(self, graph: nx.Graph, max_nodes: int = 2000) -> Tuple[np.ndarray, List[str], List]:
        """Extract comprehensive node features using multiple techniques"""
        if graph.number_of_nodes() == 0:
            return None, None, None
        
        print(f"Extracting comprehensive features from {graph.number_of_nodes()} nodes...")
        
        # Limit nodes if graph is too large
        nodes = list(graph.nodes())
        if len(nodes) > max_nodes:
            nodes = nodes[:max_nodes]
            subgraph = graph.subgraph(nodes).copy()
        else:
            subgraph = graph
            nodes = list(subgraph.nodes())
        
        all_features = []
        feature_names = []
        
        # Extract different types of features
        for feature_type, extractor in self.feature_extractors.items():
            try:
                print(f"  Extracting {feature_type} features...")
                features, names = extractor(subgraph, nodes)
                if features is not None and len(features) > 0:
                    all_features.append(features)
                    feature_names.extend([f"{feature_type}_{name}" for name in names])
                    print(f"    Added {len(names)} {feature_type} features")
            except Exception as e:
                print(f"    ⚠️ Failed to extract {feature_type} features: {e}")
        
        if not all_features:
            print("❌ No features extracted")
            return None, None, None
        
        # Combine all features
        combined_features = np.hstack(all_features)
        
        print(f"✅ Extracted {combined_features.shape[1]} total features for {len(nodes)} nodes")
        
        return combined_features, feature_names, nodes
    
    def _extract_centrality_features(self, graph: nx.Graph, nodes: List) -> Tuple[np.ndarray, List[str]]:
        """Extract centrality-based features"""
        features = []
        feature_names = []
        
        # Degree centrality
        degree_centrality = nx.degree_centrality(graph)
        features.append([degree_centrality.get(node, 0) for node in nodes])
        feature_names.append('degree_centrality')
        
        # Eigenvector centrality
        try:
            eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=1000)
            features.append([eigenvector_centrality.get(node, 0) for node in nodes])
            feature_names.append('eigenvector_centrality')
        except:
            features.append([0] * len(nodes))
            feature_names.append('eigenvector_centrality')
        
        # PageRank
        try:
            pagerank = nx.pagerank(graph, max_iter=1000)
            features.append([pagerank.get(node, 0) for node in nodes])
            feature_names.append('pagerank')
        except:
            features.append([1/len(nodes)] * len(nodes))
            feature_names.append('pagerank')
        
        # Betweenness centrality (for smaller graphs)
        if graph.number_of_nodes() < 1000:
            try:
                betweenness_centrality = nx.betweenness_centrality(graph)
                features.append([betweenness_centrality.get(node, 0) for node in nodes])
                feature_names.append('betweenness_centrality')
            except:
                features.append([0] * len(nodes))
                feature_names.append('betweenness_centrality')
        
        # Closeness centrality (for smaller graphs)
        if graph.number_of_nodes() < 1000:
            try:
                closeness_centrality = nx.closeness_centrality(graph)
                features.append([closeness_centrality.get(node, 0) for node in nodes])
                feature_names.append('closeness_centrality')
            except:
                features.append([0] * len(nodes))
                feature_names.append('closeness_centrality')
        
        # Katz centrality
        try:
            katz_centrality = nx.katz_centrality(graph, max_iter=1000)
            features.append([katz_centrality.get(node, 0) for node in nodes])
            feature_names.append('katz_centrality')
        except:
            features.append([0] * len(nodes))
            feature_names.append('katz_centrality')
        
        return np.array(features).T, feature_names
    
    def _extract_structural_features(self, graph: nx.Graph, nodes: List) -> Tuple[np.ndarray, List[str]]:
        """Extract structural features"""
        features = []
        feature_names = []
        
        # Basic degree features
        degrees = dict(graph.degree())
        features.append([degrees.get(node, 0) for node in nodes])
        feature_names.append('degree')
        
        # Clustering coefficient
        if not graph.is_directed():
            clustering = nx.clustering(graph)
            features.append([clustering.get(node, 0) for node in nodes])
            feature_names.append('clustering_coefficient')
        
        # Triangles
        if not graph.is_directed():
            try:
                triangles = nx.triangles(graph)
                features.append([triangles.get(node, 0) for node in nodes])
                feature_names.append('triangles')
            except:
                features.append([0] * len(nodes))
                feature_names.append('triangles')
        
        # Local efficiency
        try:
            local_efficiency = []
            for node in nodes:
                neighbors = list(graph.neighbors(node))
                if len(neighbors) > 1:
                    subgraph = graph.subgraph(neighbors)
                    if subgraph.number_of_edges() > 0:
                        eff = nx.global_efficiency(subgraph)
                    else:
                        eff = 0
                else:
                    eff = 0
                local_efficiency.append(eff)
            features.append(local_efficiency)
            feature_names.append('local_efficiency')
        except:
            features.append([0] * len(nodes))
            feature_names.append('local_efficiency')
        
        # Neighbor connectivity
        try:
            neighbor_connectivity = []
            for node in nodes:
                neighbors = list(graph.neighbors(node))
                if neighbors:
                    avg_neighbor_degree = np.mean([graph.degree(n) for n in neighbors])
                else:
                    avg_neighbor_degree = 0
                neighbor_connectivity.append(avg_neighbor_degree)
            features.append(neighbor_connectivity)
            feature_names.append('avg_neighbor_degree')
        except:
            features.append([0] * len(nodes))
            feature_names.append('avg_neighbor_degree')
        
        # Core number
        try:
            core_numbers = nx.core_number(graph)
            features.append([core_numbers.get(node, 0) for node in nodes])
            feature_names.append('core_number')
        except:
            features.append([0] * len(nodes))
            feature_names.append('core_number')
        
        # Eccentricity (for smaller connected graphs)
        if graph.number_of_nodes() < 500 and nx.is_connected(graph):
            try:
                eccentricity = nx.eccentricity(graph)
                features.append([eccentricity.get(node, 0) for node in nodes])
                feature_names.append('eccentricity')
            except:
                features.append([0] * len(nodes))
                feature_names.append('eccentricity')
        
        return np.array(features).T, feature_names
    
    def _extract_community_features(self, graph: nx.Graph, nodes: List) -> Tuple[np.ndarray, List[str]]:
        """Extract community-based features"""
        features = []
        feature_names = []
        
        if not COMMUNITY_AVAILABLE or graph.is_directed():
            return np.array([]).reshape(len(nodes), 0), []
        
        try:
            # Louvain community detection
            if hasattr(community_louvain, 'best_partition'):
                partition = community_louvain.best_partition(graph)
                
                # Community ID
                community_ids = [partition.get(node, -1) for node in nodes]
                features.append(community_ids)
                feature_names.append('community_id')
                
                # Community size
                community_sizes = {}
                for node, comm in partition.items():
                    if comm not in community_sizes:
                        community_sizes[comm] = 0
                    community_sizes[comm] += 1
                
                comm_sizes = [community_sizes.get(partition.get(node, -1), 0) for node in nodes]
                features.append(comm_sizes)
                feature_names.append('community_size')
                
                # Modularity contribution
                modularity_contrib = []
                for node in nodes:
                    node_comm = partition.get(node, -1)
                    contrib = 0
                    for neighbor in graph.neighbors(node):
                        if partition.get(neighbor, -1) == node_comm:
                            contrib += 1
                    modularity_contrib.append(contrib / max(1, graph.degree(node)))
                features.append(modularity_contrib)
                feature_names.append('modularity_contribution')
            
            else:
                # Use NetworkX community detection
                communities = list(nx.community.greedy_modularity_communities(graph))
                
                # Create node to community mapping
                node_to_comm = {}
                for i, community in enumerate(communities):
                    for node in community:
                        node_to_comm[node] = i
                
                # Community features
                community_ids = [node_to_comm.get(node, -1) for node in nodes]
                features.append(community_ids)
                feature_names.append('community_id')
                
                comm_sizes = [len(communities[node_to_comm.get(node, 0)]) if node in node_to_comm else 0 for node in nodes]
                features.append(comm_sizes)
                feature_names.append('community_size')
        
        except Exception as e:
            print(f"Community detection failed: {e}")
            # Fallback: use connected components
            try:
                components = list(nx.connected_components(graph))
                node_to_comp = {}
                for i, component in enumerate(components):
                    for node in component:
                        node_to_comp[node] = i
                
                comp_ids = [node_to_comp.get(node, -1) for node in nodes]
                features.append(comp_ids)
                feature_names.append('component_id')
                
                comp_sizes = [len(components[node_to_comp.get(node, 0)]) if node in node_to_comp else 0 for node in nodes]
                features.append(comp_sizes)
                feature_names.append('component_size')
            except:
                pass
        
        if not features:
            return np.array([]).reshape(len(nodes), 0), []
        
        return np.array(features).T, feature_names
    
    def _extract_spectral_features(self, graph: nx.Graph, nodes: List) -> Tuple[np.ndarray, List[str]]:
        """Extract spectral features from graph Laplacian"""
        features = []
        feature_names = []
        
        try:
            if graph.number_of_nodes() < 1000:  # Only for smaller graphs
                # Get adjacency matrix
                adj_matrix = nx.adjacency_matrix(graph, nodelist=nodes)
                
                # Compute Laplacian
                laplacian = nx.laplacian_matrix(graph, nodelist=nodes)
                
                # Compute eigenvalues and eigenvectors
                eigenvalues, eigenvectors = np.linalg.eigh(laplacian.toarray())
                
                # Use first few eigenvectors as features
                n_components = min(5, len(eigenvalues))
                for i in range(n_components):
                    features.append(eigenvectors[:, i])
                    feature_names.append(f'laplacian_eigenvector_{i}')
                
                # Fiedler vector (second smallest eigenvalue)
                if len(eigenvalues) > 1:
                    fiedler_vector = eigenvectors[:, 1]
                    features.append(fiedler_vector)
                    feature_names.append('fiedler_vector')
        
        except Exception as e:
            print(f"Spectral feature extraction failed: {e}")
        
        if not features:
            return np.array([]).reshape(len(nodes), 0), []
        
        return np.array(features).T, feature_names
    
    def _extract_motif_features(self, graph: nx.Graph, nodes: List) -> Tuple[np.ndarray, List[str]]:
        """Extract graph motif features"""
        features = []
        feature_names = []
        
        try:
            # Count triangles for each node
            if not graph.is_directed():
                triangles = nx.triangles(graph)
                features.append([triangles.get(node, 0) for node in nodes])
                feature_names.append('triangle_count')
                
                # Count squares (4-cycles)
                squares = []
                for node in nodes:
                    square_count = 0
                    neighbors = list(graph.neighbors(node))
                    for i, n1 in enumerate(neighbors):
                        for j, n2 in enumerate(neighbors[i+1:], i+1):
                            # Check if n1 and n2 have a common neighbor other than node
                            common_neighbors = set(graph.neighbors(n1)) & set(graph.neighbors(n2))
                            common_neighbors.discard(node)
                            square_count += len(common_neighbors)
                    squares.append(square_count)
                features.append(squares)
                feature_names.append('square_count')
            
            # Wedge count (2-paths)
            wedges = []
            for node in nodes:
                degree = graph.degree(node)
                wedge_count = degree * (degree - 1) // 2
                wedges.append(wedge_count)
            features.append(wedges)
            feature_names.append('wedge_count')
            
        except Exception as e:
            print(f"Motif feature extraction failed: {e}")
        
        if not features:
            return np.array([]).reshape(len(nodes), 0), []
        
        return np.array(features).T, feature_names
    
    def _extract_node2vec_features(self, graph: nx.Graph, nodes: List) -> Tuple[np.ndarray, List[str]]:
        """Extract Node2Vec embeddings"""
        if not NODE2VEC_AVAILABLE:
            return np.array([]).reshape(len(nodes), 0), []
        
        try:
            print("    Computing Node2Vec embeddings...")
            
            # Create Node2Vec model
            node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=1)
            
            # Fit model
            model = node2vec.fit(window=10, min_count=1, batch_words=4)
            
            # Get embeddings
            embeddings = []
            for node in nodes:
                if str(node) in model.wv:
                    embeddings.append(model.wv[str(node)])
                else:
                    embeddings.append(np.zeros(64))
            
            embeddings = np.array(embeddings)
            feature_names = [f'node2vec_dim_{i}' for i in range(embeddings.shape[1])]
            
            return embeddings, feature_names
        
        except Exception as e:
            print(f"Node2Vec extraction failed: {e}")
            return np.array([]).reshape(len(nodes), 0), []
    
    def generate_advanced_labels(self, graph: nx.Graph, nodes: List, task_type: str = 'centrality_classification') -> List:
        """Generate labels using advanced techniques"""
        labels = []
        
        if task_type == 'centrality_classification':
            # Multi-class centrality classification
            try:
                betweenness = nx.betweenness_centrality(graph)
                closeness = nx.closeness_centrality(graph)
                
                for node in nodes:
                    bet_cent = betweenness.get(node, 0)
                    close_cent = closeness.get(node, 0)
                    
                    # Combine centralities for classification
                    combined_score = bet_cent + close_cent
                    
                    if combined_score > np.percentile(list(betweenness.values()) + list(closeness.values()), 80):
                        labels.append('High_Central')
                    elif combined_score > np.percentile(list(betweenness.values()) + list(closeness.values()), 40):
                        labels.append('Medium_Central')
                    else:
                        labels.append('Low_Central')
            except:
                # Fallback to degree-based
                degrees = [graph.degree(node) for node in nodes]
                degree_percentiles = np.percentile(degrees, [33, 67])
                
                for node in nodes:
                    degree = graph.degree(node)
                    if degree <= degree_percentiles[0]:
                        labels.append('Low_Degree')
                    elif degree <= degree_percentiles[1]:
                        labels.append('Medium_Degree')
                    else:
                        labels.append('High_Degree')
        
        elif task_type == 'community_role':
            # Classify nodes by their role in communities
            try:
                if COMMUNITY_AVAILABLE and not graph.is_directed():
                    if hasattr(community_louvain, 'best_partition'):
                        partition = community_louvain.best_partition(graph)
                        
                        for node in nodes:
                            node_comm = partition.get(node, -1)
                            
                            # Count internal vs external connections
                            internal_edges = 0
                            external_edges = 0
                            
                            for neighbor in graph.neighbors(node):
                                if partition.get(neighbor, -1) == node_comm:
                                    internal_edges += 1
                                else:
                                    external_edges += 1
                            
                            total_edges = internal_edges + external_edges
                            if total_edges == 0:
                                labels.append('Isolated')
                            elif external_edges / total_edges > 0.5:
                                labels.append('Bridge')
                            elif internal_edges / total_edges > 0.8:
                                labels.append('Core')
                            else:
                                labels.append('Peripheral')
                    else:
                        # Fallback
                        return self.generate_advanced_labels(graph, nodes, 'centrality_classification')
                else:
                    return self.generate_advanced_labels(graph, nodes, 'centrality_classification')
            except:
                return self.generate_advanced_labels(graph, nodes, 'centrality_classification')
        
        elif task_type == 'structural_role':
            # Classify by structural equivalence
            try:
                # Use degree and clustering to define structural roles
                degrees = dict(graph.degree())
                clustering = nx.clustering(graph) if not graph.is_directed() else {}
                
                for node in nodes:
                    degree = degrees.get(node, 0)
                    clust_coeff = clustering.get(node, 0)
                    
                    if degree == 0:
                        labels.append('Isolated')
                    elif degree == 1:
                        labels.append('Pendant')
                    elif clust_coeff > 0.7:
                        labels.append('Clustered')
                    elif degree > np.percentile(list(degrees.values()), 90):
                        labels.append('Hub')
                    else:
                        labels.append('Regular')
            except:
                return self.generate_advanced_labels(graph, nodes, 'centrality_classification')
        
        else:
            # Default to centrality classification
            return self.generate_advanced_labels(graph, nodes, 'centrality_classification')
        
        return labels
    
    def train_advanced_models(self, features: np.ndarray, labels: List, use_feature_selection: bool = True) -> Dict:
        """Train advanced ML models with feature selection and cross-validation"""
        print(f"Training advanced models on {len(features)} samples with {features.shape[1]} features...")
        
        # Encode string labels to numeric if needed
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        
        # Store label mapping for later reference
        label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        print(f"Label mapping: {label_mapping}")
        
        # Check data quality
        unique_labels, counts = np.unique(labels_encoded, return_counts=True)
        min_samples = min(counts)
        
        if len(features) < 50:
            print("⚠️ Limited samples for reliable training")
        
        if min_samples < 3:
            print("⚠️ Some classes have very few samples. Results may be unreliable.")
        
        # Feature selection
        if use_feature_selection and features.shape[1] > 20:
            print("Performing feature selection...")
            try:
                # Select top k features
                k = min(20, features.shape[1])
                selector = SelectKBest(score_func=f_classif, k=k)
                features_selected = selector.fit_transform(features, labels_encoded)
                selected_features = selector.get_support(indices=True)
                print(f"Selected {len(selected_features)} most informative features")
            except:
                features_selected = features
                print("Feature selection failed, using all features")
        else:
            features_selected = features
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                features_selected, labels_encoded, test_size=0.25, random_state=42, stratify=labels_encoded
            )
        except:
            X_train, X_test, y_train, y_test = train_test_split(
                features_selected, labels_encoded, test_size=0.25, random_state=42
            )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        
        # Train traditional models
        for model_name, model in self.traditional_models.items():
            try:
                print(f"Training {model_name}...")
                
                # Cross-validation
                if len(X_train) > 30:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                else:
                    cv_mean = cv_std = 0
                
                # Train and evaluate
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                
                # ROC AUC for binary classification
                try:
                    if len(unique_labels) == 2 and hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_test_scaled)[:, 1]
                        auc_score = roc_auc_score(y_test, y_proba)
                    else:
                        auc_score = None
                except:
                    auc_score = None
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'auc_score': auc_score,
                    'classification_report': report,
                    'y_test': y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test),
                    'y_pred': y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred),
                    'y_test_labels': label_encoder.inverse_transform(y_test).tolist(),
                    'y_pred_labels': label_encoder.inverse_transform(y_pred).tolist(),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                    'label_mapping': label_mapping
                }
                
                print(f"{model_name} - Accuracy: {accuracy:.3f}, CV: {cv_mean:.3f}±{cv_std:.3f}")
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                results[model_name] = None
        
        # Train Graph Neural Networks if available
        if TORCH_GEOMETRIC_AVAILABLE:
            try:
                gnn_results = self._train_graph_neural_networks(features, labels, X_train, X_test, y_train, y_test)
                results.update(gnn_results)
            except Exception as e:
                print(f"GNN training failed: {e}")
        
        return results
    
    def _train_graph_neural_networks(self, features, labels, X_train, X_test, y_train, y_test):
        """Train Graph Neural Networks"""
        print("Training Graph Neural Networks...")
        
        # This is a simplified GNN implementation
        # In practice, you'd need the graph structure for proper GNN training
        
        gnn_results = {}
        
        try:
            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train)
            X_test_tensor = torch.FloatTensor(X_test)
            
            # Encode labels
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)
            
            y_train_tensor = torch.LongTensor(y_train_encoded)
            y_test_tensor = torch.LongTensor(y_test_encoded)
            
            # Simple MLP as baseline "GNN"
            class SimpleMLP(nn.Module):
                def __init__(self, input_dim, hidden_dim, output_dim):
                    super(SimpleMLP, self).__init__()
                    self.layers = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(hidden_dim // 2, output_dim)
                    )
                
                def forward(self, x):
                    return self.layers(x)
            
            # Train MLP
            input_dim = X_train.shape[1]
            hidden_dim = min(128, input_dim * 2)
            output_dim = len(np.unique(y_train))
            
            model = SimpleMLP(input_dim, hidden_dim, output_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            model.train()
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                _, predicted = torch.max(test_outputs.data, 1)
                
                accuracy = (predicted == y_test_tensor).float().mean().item()
                
                gnn_results['Deep MLP'] = {
                    'accuracy': accuracy,
                    'cv_mean': 0,  # Not implemented for GNNs
                    'cv_std': 0,
                    'auc_score': None,
                    'classification_report': classification_report(y_test_encoded, predicted.numpy(), output_dict=True),
                    'y_test': y_test,
                    'y_pred': label_encoder.inverse_transform(predicted.numpy()),
                    'confusion_matrix': confusion_matrix(y_test_encoded, predicted.numpy()).tolist()
                }
                
                print(f"Deep MLP - Accuracy: {accuracy:.3f}")
        
        except Exception as e:
            print(f"GNN training error: {e}")
        
        return gnn_results
    
    def compare_advanced_datasets(self, original_graph: nx.Graph, cleaned_graph: nx.Graph, 
                                task_type: str = 'centrality_classification', max_nodes: int = 2000) -> Dict:
        """Compare ML performance with advanced techniques"""
        print("=== Advanced Graph Dataset Comparison ===")
        
        # Extract features from original graph
        print("\n1. Processing Original Graph...")
        orig_features, orig_feature_names, orig_node_ids = self.extract_comprehensive_features(original_graph, max_nodes)
        
        if orig_features is None:
            print("❌ No features extracted from original graph")
            return None
        
        orig_labels = self.generate_advanced_labels(original_graph, orig_node_ids, task_type)
        
        # Extract features from cleaned graph
        print("\n2. Processing Cleaned Graph...")
        clean_features, clean_feature_names, clean_node_ids = self.extract_comprehensive_features(cleaned_graph, max_nodes)
        
        if clean_features is None:
            print("❌ No features extracted from cleaned graph")
            return None
        
        clean_labels = self.generate_advanced_labels(cleaned_graph, clean_node_ids, task_type)
        
        # Train models on original graph
        print("\n3. Training Advanced Models on Original Graph...")
        original_results = self.train_advanced_models(orig_features, orig_labels)
        
        # Train models on cleaned graph
        print("\n4. Training Advanced Models on Cleaned Graph...")
        cleaned_results = self.train_advanced_models(clean_features, clean_labels)
        
        if not original_results or not cleaned_results:
            print("❌ Model training failed")
            return None
        
        # Compare results
        print("\n5. Comparing Advanced Results...")
        comparison = self._compare_advanced_results(original_results, cleaned_results)
        
        return {
            'original_results': original_results,
            'cleaned_results': cleaned_results,
            'comparison': comparison,
            'original_samples': len(orig_features),
            'cleaned_samples': len(clean_features),
            'original_features': orig_features.shape[1],
            'cleaned_features': clean_features.shape[1],
            'task_type': task_type,
            'feature_names': orig_feature_names
        }
    
    def _compare_advanced_results(self, original_results: Dict, cleaned_results: Dict) -> Dict:
        """Compare advanced results with detailed metrics"""
        comparison = {}
        
        all_models = set(original_results.keys()) | set(cleaned_results.keys())
        
        for model_name in all_models:
            if (model_name in original_results and original_results[model_name] and
                model_name in cleaned_results and cleaned_results[model_name]):
                
                orig_result = original_results[model_name]
                clean_result = cleaned_results[model_name]
                
                orig_acc = orig_result['accuracy']
                clean_acc = clean_result['accuracy']
                improvement = clean_acc - orig_acc
                improvement_pct = (improvement / orig_acc) * 100 if orig_acc > 0 else 0
                
                # Cross-validation comparison
                orig_cv = orig_result.get('cv_mean', 0)
                clean_cv = clean_result.get('cv_mean', 0)
                cv_improvement = clean_cv - orig_cv
                
                # AUC comparison
                orig_auc = orig_result.get('auc_score')
                clean_auc = clean_result.get('auc_score')
                auc_improvement = None
                if orig_auc is not None and clean_auc is not None:
                    auc_improvement = clean_auc - orig_auc
                
                comparison[model_name] = {
                    'original_accuracy': orig_acc,
                    'cleaned_accuracy': clean_acc,
                    'accuracy_improvement': improvement,
                    'accuracy_improvement_percentage': improvement_pct,
                    'original_cv_score': orig_cv,
                    'cleaned_cv_score': clean_cv,
                    'cv_improvement': cv_improvement,
                    'original_auc': orig_auc,
                    'cleaned_auc': clean_auc,
                    'auc_improvement': auc_improvement
                }
        
        return comparison
    
    def create_advanced_visualization(self, comparison_results: Dict, save_path: str) -> str:
        """Create comprehensive visualization of advanced results"""
        if not comparison_results or 'comparison' not in comparison_results:
            return None
        
        comparison = comparison_results['comparison']
        
        fig, axes = plt.subplots(3, 2, figsize=(18, 15))
        axes = axes.flatten()
        
        # Main title
        task_type = comparison_results.get('task_type', 'Node Classification')
        fig.suptitle(f'DataDome Advanced Graph ML: {task_type.replace("_", " ").title()} Performance', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        models = list(comparison.keys())
        
        # 1. Accuracy Comparison
        original_accs = [comparison[model]['original_accuracy'] for model in models]
        cleaned_accs = [comparison[model]['cleaned_accuracy'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = axes[0].bar(x - width/2, original_accs, width, label='Original Graph', 
                           color='#FF6B6B', alpha=0.8)
        bars2 = axes[0].bar(x + width/2, cleaned_accs, width, label='Cleaned Graph', 
                           color='#4ECDC4', alpha=0.8)
        
        axes[0].set_title('Model Accuracy Comparison', fontsize=14, pad=20)
        axes[0].set_ylabel('Accuracy')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Cross-Validation Scores
        orig_cv_scores = [comparison[model]['original_cv_score'] for model in models]
        clean_cv_scores = [comparison[model]['cleaned_cv_score'] for model in models]
        
        bars1 = axes[1].bar(x - width/2, orig_cv_scores, width, label='Original Graph', 
                           color='#FF6B6B', alpha=0.8)
        bars2 = axes[1].bar(x + width/2, clean_cv_scores, width, label='Cleaned Graph', 
                           color='#4ECDC4', alpha=0.8)
        
        axes[1].set_title('Cross-Validation Scores', fontsize=14, pad=20)
        axes[1].set_ylabel('CV Accuracy')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models, rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        # 3. Improvement Percentage
        improvements = [comparison[model]['accuracy_improvement_percentage'] for model in models]
        colors = ['#4ECDC4' if imp > 0 else '#FF6B6B' for imp in improvements]
        
        bars = axes[2].bar(models, improvements, color=colors, alpha=0.8)
        axes[2].set_title('Accuracy Improvement (%)', fontsize=14, pad=20)
        axes[2].set_ylabel('Improvement Percentage')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(axis='y', alpha=0.3)
        axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2., 
                        height + (0.5 if height > 0 else -1),
                        f'{imp:.1f}%', ha='center', 
                        va='bottom' if height > 0 else 'top', fontsize=10)
        
        # 4. Feature Count Comparison
        feature_counts = [
            comparison_results['original_features'],
            comparison_results['cleaned_features']
        ]
        feature_labels = ['Original\nGraph', 'Cleaned\nGraph']
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = axes[3].bar(feature_labels, feature_counts, color=colors, alpha=0.8)
        axes[3].set_title('Feature Count Comparison', fontsize=14, pad=20)
        axes[3].set_ylabel('Number of Features')
        axes[3].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, count in zip(bars, feature_counts):
            height = bar.get_height()
            axes[3].text(bar.get_x() + bar.get_width()/2., height + max(feature_counts)*0.01,
                        f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 5. AUC Scores (if available)
        auc_models = [m for m in models if comparison[m]['original_auc'] is not None]
        if auc_models:
            orig_aucs = [comparison[model]['original_auc'] for model in auc_models]
            clean_aucs = [comparison[model]['cleaned_auc'] for model in auc_models]
            
            x_auc = np.arange(len(auc_models))
            bars1 = axes[4].bar(x_auc - width/2, orig_aucs, width, label='Original Graph', 
                               color='#FF6B6B', alpha=0.8)
            bars2 = axes[4].bar(x_auc + width/2, clean_aucs, width, label='Cleaned Graph', 
                               color='#4ECDC4', alpha=0.8)
            
            axes[4].set_title('AUC Scores', fontsize=14, pad=20)
            axes[4].set_ylabel('AUC Score')
            axes[4].set_xticks(x_auc)
            axes[4].set_xticklabels(auc_models, rotation=45, ha='right')
            axes[4].legend()
            axes[4].grid(axis='y', alpha=0.3)
        else:
            axes[4].text(0.5, 0.5, 'AUC Scores\nNot Available\n(Multi-class task)', 
                        ha='center', va='center', transform=axes[4].transAxes, fontsize=12)
            axes[4].set_title('AUC Scores', fontsize=14, pad=20)
        
        # 6. Performance Summary
        avg_improvement = np.mean(improvements)
        best_model = models[np.argmax(improvements)]
        best_improvement = max(improvements)
        
        axes[5].text(0.1, 0.9, 'Advanced ML Performance Summary:', fontsize=14, fontweight='bold', 
                    transform=axes[5].transAxes)
        
        summary_text = f"""
Models Evaluated: {len(models)}
Average Improvement: {avg_improvement:.1f}%

Best Model: {best_model}
Best Improvement: {best_improvement:.1f}%

Original Samples: {comparison_results['original_samples']:,}
Cleaned Samples: {comparison_results['cleaned_samples']:,}

Original Features: {comparison_results['original_features']}
Cleaned Features: {comparison_results['cleaned_features']}

Task: {task_type.replace('_', ' ').title()}
        """
        
        axes[5].text(0.1, 0.05, summary_text, fontsize=11, transform=axes[5].transAxes,
                    verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor='lightgray', alpha=0.5))
        axes[5].axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, hspace=0.4, wspace=0.3)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Advanced graph ML visualization saved to: {save_path}")
        return save_path