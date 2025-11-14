import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import warnings
from typing import Dict, List, Tuple, Optional, Any
import logging

# Neo4j imports
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("⚠️ Neo4j driver not installed. Install with: pip install neo4j")

# Graph ML imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv
    from torch_geometric.utils import from_networkx
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("⚠️ PyTorch Geometric not installed. Install with: pip install torch torch-geometric")

# Community detection
try:
    import community as community_louvain
    COMMUNITY_AVAILABLE = True
except ImportError:
    try:
        from networkx.algorithms import community
        COMMUNITY_AVAILABLE = True
    except ImportError:
        COMMUNITY_AVAILABLE = False

warnings.filterwarnings('ignore')

class Neo4jGraphProcessor:
    """Advanced graph processing with Neo4j integration and modern ML techniques"""
    
    def __init__(self, neo4j_uri=None, neo4j_user=None, neo4j_password=None):
        self.graph = None
        self.original_graph = None
        self.neo4j_driver = None
        self.cleaning_stats = {
            'original_nodes': 0,
            'original_edges': 0,
            'isolated_nodes_removed': 0,
            'duplicate_edges_removed': 0,
            'self_loops_removed': 0,
            'low_degree_nodes_removed': 0,
            'disconnected_components_removed': 0,
            'anomalous_nodes_removed': 0,
            'bridge_edges_preserved': 0,
            'final_nodes': 0,
            'final_edges': 0
        }
        
        # Initialize Neo4j connection if credentials provided
        if neo4j_uri and neo4j_user and neo4j_password and NEO4J_AVAILABLE:
            try:
                self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
                print("✅ Connected to Neo4j database")
            except Exception as e:
                print(f"⚠️ Failed to connect to Neo4j: {e}")
                self.neo4j_driver = None
    
    def __del__(self):
        if self.neo4j_driver:
            self.neo4j_driver.close()
    
    def load_graph_advanced(self, file_path: str, directed: bool = False, weighted: bool = False, **kwargs) -> bool:
        """Advanced graph loading with multiple format support and validation"""
        try:
            print(f"Loading graph from {file_path}...")
            
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Determine graph type
            create_using = nx.DiGraph() if directed else nx.Graph()
            
            if file_ext == '.csv':
                self.graph = self._load_from_csv(file_path, directed=directed, weighted=weighted)
            elif file_ext == '.gml':
                self.graph = nx.read_gml(file_path)
            elif file_ext == '.graphml':
                self.graph = nx.read_graphml(file_path)
            elif file_ext == '.gexf':
                self.graph = nx.read_gexf(file_path)
            elif file_ext in ['.txt', '.edgelist']:
                # read_edgelist doesn't accept 'directed' parameter, use create_using instead
                self.graph = nx.read_edgelist(file_path, create_using=create_using)
            elif file_ext == '.json':
                self.graph = self._load_from_json(file_path, directed=directed)
            else:
                # Try to auto-detect format
                self.graph = self._auto_detect_format(file_path, directed=directed, weighted=weighted)
            
            if self.graph is None:
                return False
            
            # Store original copy
            self.original_graph = self.graph.copy()
            
            # Update stats
            self.cleaning_stats['original_nodes'] = self.graph.number_of_nodes()
            self.cleaning_stats['original_edges'] = self.graph.number_of_edges()
            
            # Validate graph
            validation_results = self._validate_graph_structure()
            
            print(f"✅ Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            if validation_results['warnings']:
                print("⚠️ Graph validation warnings:")
                for warning in validation_results['warnings']:
                    print(f"   - {warning}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading graph: {e}")
            return False
    
    def _load_from_csv(self, file_path: str, **kwargs) -> nx.Graph:
        """Load graph from CSV with intelligent column detection"""
        df = pd.read_csv(file_path)
        
        # Auto-detect source and target columns
        possible_source_cols = ['source', 'src', 'from', 'node1', 'u', 'start']
        possible_target_cols = ['target', 'dst', 'to', 'node2', 'v', 'end']
        possible_weight_cols = ['weight', 'w', 'cost', 'distance', 'strength']
        
        source_col = None
        target_col = None
        weight_col = None
        
        # Find source column
        for col in df.columns:
            if col.lower() in possible_source_cols:
                source_col = col
                break
        
        # Find target column
        for col in df.columns:
            if col.lower() in possible_target_cols:
                target_col = col
                break
        
        # Find weight column
        for col in df.columns:
            if col.lower() in possible_weight_cols:
                weight_col = col
                break
        
        # Fallback to first two columns
        if source_col is None or target_col is None:
            if len(df.columns) >= 2:
                source_col = df.columns[0]
                target_col = df.columns[1]
                if len(df.columns) >= 3:
                    weight_col = df.columns[2]
            else:
                raise ValueError("CSV must have at least 2 columns")
        
        # Create graph
        directed = kwargs.get('directed', False)
        G = nx.DiGraph() if directed else nx.Graph()
        
        # Add edges
        for _, row in df.iterrows():
            source = row[source_col]
            target = row[target_col]
            
            if weight_col and pd.notna(row[weight_col]):
                G.add_edge(source, target, weight=float(row[weight_col]))
            else:
                G.add_edge(source, target)
        
        return G
    
    def _load_from_json(self, file_path: str, directed: bool = False) -> nx.Graph:
        """Load graph from JSON format"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if 'nodes' in data and 'links' in data:
            # D3.js format
            G = nx.DiGraph() if directed else nx.Graph()
            
            # Add nodes
            for node in data['nodes']:
                node_id = node.get('id', node.get('name'))
                G.add_node(node_id, **{k: v for k, v in node.items() if k not in ['id', 'name']})
            
            # Add edges
            for link in data['links']:
                source = link.get('source')
                target = link.get('target')
                weight = link.get('weight', link.get('value', 1))
                G.add_edge(source, target, weight=weight)
            
            return G
        else:
            # NetworkX JSON format
            return nx.node_link_graph(data)
    
    def _auto_detect_format(self, file_path: str, directed: bool = False, weighted: bool = False) -> nx.Graph:
        """Auto-detect file format and load accordingly"""
        create_using = nx.DiGraph() if directed else nx.Graph()
        
        try:
            # Try edge list first
            return nx.read_edgelist(file_path, create_using=create_using)
        except:
            try:
                # Try as CSV
                return self._load_from_csv(file_path, directed=directed, weighted=weighted)
            except:
                raise ValueError(f"Could not auto-detect format for {file_path}")
    
    def _validate_graph_structure(self) -> Dict[str, Any]:
        """Validate graph structure and identify potential issues"""
        warnings = []
        
        if self.graph.number_of_nodes() == 0:
            warnings.append("Graph has no nodes")
        
        if self.graph.number_of_edges() == 0:
            warnings.append("Graph has no edges")
        
        # Check for isolated nodes
        isolated = [n for n in self.graph.nodes() if self.graph.degree(n) == 0]
        if isolated:
            warnings.append(f"Found {len(isolated)} isolated nodes")
        
        # Check for self loops
        self_loops = list(nx.selfloop_edges(self.graph))
        if self_loops:
            warnings.append(f"Found {len(self_loops)} self loops")
        
        # Check for very high degree nodes (potential hubs or errors)
        degrees = dict(self.graph.degree())
        if degrees:
            max_degree = max(degrees.values())
            avg_degree = sum(degrees.values()) / len(degrees)
            if max_degree > avg_degree * 10:
                warnings.append(f"Found potential hub/anomaly: node with degree {max_degree} (avg: {avg_degree:.1f})")
        
        return {'warnings': warnings}
    
    def advanced_graph_cleaning(self, 
                              remove_isolated_nodes: bool = True,
                              remove_self_loops: bool = True,
                              remove_duplicate_edges: bool = True,
                              min_degree_threshold: int = 1,
                              max_degree_threshold: Optional[int] = None,
                              keep_largest_component: bool = True,
                              min_component_size: int = 5,
                              remove_bridges: bool = False,
                              anomaly_detection: bool = True,
                              preserve_important_nodes: bool = True) -> bool:
        """Advanced graph cleaning with modern techniques"""
        
        if self.graph is None:
            print("❌ No graph loaded")
            return False
        
        print("🧹 Starting advanced graph cleaning...")
        
        # 1. Anomaly detection
        if anomaly_detection:
            anomalous_nodes = self._detect_anomalous_nodes()
            if anomalous_nodes and not preserve_important_nodes:
                self.graph.remove_nodes_from(anomalous_nodes)
                self.cleaning_stats['anomalous_nodes_removed'] = len(anomalous_nodes)
                print(f"Removed {len(anomalous_nodes)} anomalous nodes")
        
        # 2. Remove self loops
        if remove_self_loops:
            self_loops = list(nx.selfloop_edges(self.graph))
            self.graph.remove_edges_from(self_loops)
            self.cleaning_stats['self_loops_removed'] = len(self_loops)
            print(f"Removed {len(self_loops)} self loops")
        
        # 3. Remove duplicate edges
        if remove_duplicate_edges:
            if isinstance(self.graph, (nx.MultiGraph, nx.MultiDiGraph)):
                original_edges = self.graph.number_of_edges()
                self.graph = nx.Graph(self.graph) if not self.graph.is_directed() else nx.DiGraph(self.graph)
                duplicates_removed = original_edges - self.graph.number_of_edges()
                self.cleaning_stats['duplicate_edges_removed'] = duplicates_removed
                print(f"Removed {duplicates_removed} duplicate edges")
        
        # 4. Identify and preserve important structures
        important_nodes = set()
        if preserve_important_nodes:
            important_nodes = self._identify_important_nodes()
            print(f"Identified {len(important_nodes)} important nodes to preserve")
        
        # 5. Remove nodes by degree thresholds
        nodes_to_remove = []
        
        # Low degree nodes
        if min_degree_threshold > 0:
            low_degree_nodes = [n for n in self.graph.nodes() 
                              if self.graph.degree(n) < min_degree_threshold 
                              and n not in important_nodes]
            nodes_to_remove.extend(low_degree_nodes)
            self.cleaning_stats['low_degree_nodes_removed'] = len(low_degree_nodes)
        
        # High degree nodes (potential anomalies)
        if max_degree_threshold:
            high_degree_nodes = [n for n in self.graph.nodes() 
                               if self.graph.degree(n) > max_degree_threshold 
                               and n not in important_nodes]
            nodes_to_remove.extend(high_degree_nodes)
        
        # Remove nodes
        if nodes_to_remove:
            self.graph.remove_nodes_from(nodes_to_remove)
            print(f"Removed {len(nodes_to_remove)} nodes based on degree thresholds")
        
        # 6. Remove isolated nodes
        if remove_isolated_nodes:
            isolated_nodes = [n for n in self.graph.nodes() if self.graph.degree(n) == 0]
            self.graph.remove_nodes_from(isolated_nodes)
            self.cleaning_stats['isolated_nodes_removed'] = len(isolated_nodes)
            print(f"Removed {len(isolated_nodes)} isolated nodes")
        
        # 7. Handle connected components
        if keep_largest_component:
            components = list(nx.connected_components(self.graph) if not self.graph.is_directed() 
                            else nx.weakly_connected_components(self.graph))
            
            if len(components) > 1:
                # Keep largest component and components above minimum size
                largest_component = max(components, key=len)
                nodes_to_remove = []
                
                for component in components:
                    if component != largest_component and len(component) < min_component_size:
                        nodes_to_remove.extend(component)
                
                self.graph.remove_nodes_from(nodes_to_remove)
                self.cleaning_stats['disconnected_components_removed'] = len(nodes_to_remove)
                print(f"Removed {len(nodes_to_remove)} nodes from small components")
        
        # 8. Bridge preservation
        if not remove_bridges:
            bridges = list(nx.bridges(self.graph)) if not self.graph.is_directed() else []
            self.cleaning_stats['bridge_edges_preserved'] = len(bridges)
            if bridges:
                print(f"Preserved {len(bridges)} bridge edges")
        
        # Update final stats
        self.cleaning_stats['final_nodes'] = self.graph.number_of_nodes()
        self.cleaning_stats['final_edges'] = self.graph.number_of_edges()
        
        print(f"✅ Advanced graph cleaning completed:")
        print(f"  Nodes: {self.cleaning_stats['original_nodes']} → {self.cleaning_stats['final_nodes']}")
        print(f"  Edges: {self.cleaning_stats['original_edges']} → {self.cleaning_stats['final_edges']}")
        
        return True
    
    def _detect_anomalous_nodes(self) -> List:
        """Detect anomalous nodes using statistical methods"""
        anomalous_nodes = []
        
        # Degree-based anomaly detection
        degrees = [self.graph.degree(n) for n in self.graph.nodes()]
        if degrees:
            q75, q25 = np.percentile(degrees, [75, 25])
            iqr = q75 - q25
            upper_bound = q75 + 1.5 * iqr
            
            for node in self.graph.nodes():
                if self.graph.degree(node) > upper_bound:
                    anomalous_nodes.append(node)
        
        return anomalous_nodes
    
    def _identify_important_nodes(self) -> set:
        """Identify structurally important nodes to preserve"""
        important_nodes = set()
        
        try:
            # High centrality nodes
            if self.graph.number_of_nodes() < 1000:  # Only for smaller graphs
                betweenness = nx.betweenness_centrality(self.graph)
                closeness = nx.closeness_centrality(self.graph)
                
                # Top 10% by centrality
                n_important = max(1, self.graph.number_of_nodes() // 10)
                
                top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:n_important]
                top_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:n_important]
                
                important_nodes.update([node for node, _ in top_betweenness])
                important_nodes.update([node for node, _ in top_closeness])
            
            # Articulation points (cut vertices)
            if not self.graph.is_directed():
                articulation_points = list(nx.articulation_points(self.graph))
                important_nodes.update(articulation_points)
            
        except Exception as e:
            print(f"Warning: Could not identify important nodes: {e}")
        
        return important_nodes
    
    def export_to_neo4j(self, clear_existing: bool = True) -> bool:
        """Export graph to Neo4j database"""
        if not self.neo4j_driver:
            print("❌ No Neo4j connection available")
            return False
        
        if self.graph is None:
            print("❌ No graph to export")
            return False
        
        try:
            with self.neo4j_driver.session() as session:
                # Clear existing data if requested
                if clear_existing:
                    session.run("MATCH (n) DETACH DELETE n")
                    print("Cleared existing Neo4j data")
                
                # Create nodes
                print("Creating nodes in Neo4j...")
                for node in self.graph.nodes(data=True):
                    node_id, attributes = node
                    
                    # Prepare node properties
                    props = {"id": str(node_id)}
                    props.update({k: v for k, v in attributes.items() if isinstance(v, (str, int, float, bool))})
                    
                    session.run(
                        "CREATE (n:Node $props)",
                        props=props
                    )
                
                # Create relationships
                print("Creating relationships in Neo4j...")
                for edge in self.graph.edges(data=True):
                    source, target, attributes = edge
                    
                    # Prepare edge properties
                    props = {k: v for k, v in attributes.items() if isinstance(v, (str, int, float, bool))}
                    
                    session.run(
                        "MATCH (a:Node {id: $source}), (b:Node {id: $target}) "
                        "CREATE (a)-[r:CONNECTED $props]->(b)",
                        source=str(source), target=str(target), props=props
                    )
                
                print(f"✅ Exported {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges to Neo4j")
                return True
                
        except Exception as e:
            print(f"❌ Error exporting to Neo4j: {e}")
            return False
    
    def load_from_neo4j(self, node_label: str = "Node", relationship_type: str = "CONNECTED") -> bool:
        """Load graph from Neo4j database"""
        if not self.neo4j_driver:
            print("❌ No Neo4j connection available")
            return False
        
        try:
            with self.neo4j_driver.session() as session:
                # Load nodes
                nodes_result = session.run(f"MATCH (n:{node_label}) RETURN n")
                
                # Load relationships
                edges_result = session.run(
                    f"MATCH (a:{node_label})-[r:{relationship_type}]->(b:{node_label}) "
                    "RETURN a.id as source, b.id as target, r"
                )
                
                # Create graph
                self.graph = nx.Graph()
                
                # Add nodes
                for record in nodes_result:
                    node = record["n"]
                    node_id = node.get("id")
                    properties = dict(node)
                    self.graph.add_node(node_id, **properties)
                
                # Add edges
                for record in edges_result:
                    source = record["source"]
                    target = record["target"]
                    relationship = record["r"]
                    properties = dict(relationship) if relationship else {}
                    self.graph.add_edge(source, target, **properties)
                
                self.original_graph = self.graph.copy()
                
                # Update stats
                self.cleaning_stats['original_nodes'] = self.graph.number_of_nodes()
                self.cleaning_stats['original_edges'] = self.graph.number_of_edges()
                
                print(f"✅ Loaded {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges from Neo4j")
                return True
                
        except Exception as e:
            print(f"❌ Error loading from Neo4j: {e}")
            return False
    
    def run_neo4j_analytics(self) -> Dict[str, Any]:
        """Run advanced analytics using Neo4j's built-in algorithms"""
        if not self.neo4j_driver:
            print("❌ No Neo4j connection available")
            return {}
        
        analytics_results = {}
        
        try:
            with self.neo4j_driver.session() as session:
                # PageRank
                try:
                    session.run("CALL gds.pageRank.write('myGraph', {writeProperty: 'pagerank'})")
                    pagerank_result = session.run("MATCH (n) RETURN avg(n.pagerank) as avg_pagerank")
                    analytics_results['avg_pagerank'] = pagerank_result.single()['avg_pagerank']
                except:
                    print("⚠️ PageRank algorithm not available")
                
                # Community detection
                try:
                    session.run("CALL gds.louvain.write('myGraph', {writeProperty: 'community'})")
                    community_result = session.run("MATCH (n) RETURN count(DISTINCT n.community) as num_communities")
                    analytics_results['num_communities'] = community_result.single()['num_communities']
                except:
                    print("⚠️ Louvain community detection not available")
                
                # Centrality measures
                try:
                    betweenness_result = session.run(
                        "CALL gds.betweenness.stream('myGraph') "
                        "YIELD nodeId, score "
                        "RETURN avg(score) as avg_betweenness"
                    )
                    analytics_results['avg_betweenness'] = betweenness_result.single()['avg_betweenness']
                except:
                    print("⚠️ Betweenness centrality not available")
                
        except Exception as e:
            print(f"⚠️ Error running Neo4j analytics: {e}")
        
        return analytics_results
    
    def calculate_advanced_metrics(self) -> Dict[str, Any]:
        """Calculate advanced graph metrics using modern techniques"""
        if self.graph is None:
            return {}
        
        metrics = {}
        
        try:
            # Basic metrics
            metrics['nodes'] = self.graph.number_of_nodes()
            metrics['edges'] = self.graph.number_of_edges()
            metrics['density'] = nx.density(self.graph)
            
            # Degree distribution analysis
            degrees = [self.graph.degree(n) for n in self.graph.nodes()]
            if degrees:
                metrics['degree_mean'] = np.mean(degrees)
                metrics['degree_std'] = np.std(degrees)
                metrics['degree_skewness'] = self._calculate_skewness(degrees)
                metrics['degree_gini'] = self._calculate_gini_coefficient(degrees)
            
            # Connectivity and components
            if self.graph.is_directed():
                metrics['is_weakly_connected'] = nx.is_weakly_connected(self.graph)
                metrics['is_strongly_connected'] = nx.is_strongly_connected(self.graph)
                metrics['num_weakly_connected_components'] = nx.number_weakly_connected_components(self.graph)
                metrics['num_strongly_connected_components'] = nx.number_strongly_connected_components(self.graph)
            else:
                metrics['is_connected'] = nx.is_connected(self.graph)
                metrics['num_connected_components'] = nx.number_connected_components(self.graph)
            
            # Clustering and transitivity
            if not self.graph.is_directed():
                metrics['average_clustering'] = nx.average_clustering(self.graph)
                metrics['transitivity'] = nx.transitivity(self.graph)
            
            # Small-world properties
            if metrics.get('is_connected', False) and self.graph.number_of_nodes() < 1000:
                try:
                    metrics['average_shortest_path_length'] = nx.average_shortest_path_length(self.graph)
                    metrics['diameter'] = nx.diameter(self.graph)
                    metrics['radius'] = nx.radius(self.graph)
                    
                    # Small-world coefficient
                    random_graph = nx.erdos_renyi_graph(self.graph.number_of_nodes(), 
                                                      self.graph.number_of_edges() / (self.graph.number_of_nodes() * (self.graph.number_of_nodes() - 1) / 2))
                    random_clustering = nx.average_clustering(random_graph)
                    random_path_length = nx.average_shortest_path_length(random_graph)
                    
                    if random_clustering > 0 and random_path_length > 0:
                        metrics['small_world_coefficient'] = (metrics['average_clustering'] / random_clustering) / (metrics['average_shortest_path_length'] / random_path_length)
                except:
                    pass
            
            # Centrality measures (for smaller graphs)
            if self.graph.number_of_nodes() < 1000:
                try:
                    betweenness = nx.betweenness_centrality(self.graph)
                    closeness = nx.closeness_centrality(self.graph)
                    eigenvector = nx.eigenvector_centrality(self.graph, max_iter=1000)
                    
                    metrics['avg_betweenness_centrality'] = np.mean(list(betweenness.values()))
                    metrics['avg_closeness_centrality'] = np.mean(list(closeness.values()))
                    metrics['avg_eigenvector_centrality'] = np.mean(list(eigenvector.values()))
                    
                    # Centralization measures
                    metrics['betweenness_centralization'] = self._calculate_centralization(betweenness)
                    metrics['closeness_centralization'] = self._calculate_centralization(closeness)
                    
                except:
                    pass
            
            # Community structure
            if COMMUNITY_AVAILABLE and not self.graph.is_directed() and self.graph.number_of_nodes() > 10:
                try:
                    if hasattr(community_louvain, 'best_partition'):
                        partition = community_louvain.best_partition(self.graph)
                        metrics['num_communities_louvain'] = len(set(partition.values()))
                        metrics['modularity_louvain'] = community_louvain.modularity(partition, self.graph)
                    else:
                        # Use NetworkX community detection
                        communities = nx.community.greedy_modularity_communities(self.graph)
                        metrics['num_communities_greedy'] = len(communities)
                        metrics['modularity_greedy'] = nx.community.modularity(self.graph, communities)
                except:
                    pass
            
            # Assortativity
            try:
                metrics['degree_assortativity'] = nx.degree_assortativity_coefficient(self.graph)
            except:
                pass
            
            # Rich club coefficient
            try:
                if self.graph.number_of_nodes() < 500:
                    rich_club = nx.rich_club_coefficient(self.graph)
                    if rich_club:
                        metrics['max_rich_club_coefficient'] = max(rich_club.values())
            except:
                pass
            
        except Exception as e:
            print(f"Error calculating advanced metrics: {e}")
        
        return metrics
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of a distribution"""
        if len(data) < 3:
            return 0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0
        
        skewness = np.mean([((x - mean) / std) ** 3 for x in data])
        return skewness
    
    def _calculate_gini_coefficient(self, data: List[float]) -> float:
        """Calculate Gini coefficient for inequality measurement"""
        if len(data) == 0:
            return 0
        
        sorted_data = sorted(data)
        n = len(sorted_data)
        cumsum = np.cumsum(sorted_data)
        
        return (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(cumsum))) / (n * sum(sorted_data))
    
    def _calculate_centralization(self, centrality_dict: Dict) -> float:
        """Calculate network centralization"""
        if not centrality_dict:
            return 0
        
        values = list(centrality_dict.values())
        max_centrality = max(values)
        sum_diff = sum(max_centrality - c for c in values)
        
        n = len(values)
        max_possible_sum = (n - 1) * (n - 2) if n > 2 else 1
        
        return sum_diff / max_possible_sum if max_possible_sum > 0 else 0
    
    def export_advanced_formats(self, output_dir: str = "output") -> Dict[str, str]:
        """Export graph in multiple advanced formats"""
        if self.graph is None:
            return {}
        
        os.makedirs(output_dir, exist_ok=True)
        exported_files = {}
        
        try:
            # NetworkX formats
            formats = {
                'graphml': lambda g, p: nx.write_graphml(g, p),
                'gexf': lambda g, p: nx.write_gexf(g, p),
                'gml': lambda g, p: nx.write_gml(g, p),
                'edgelist': lambda g, p: nx.write_edgelist(g, p),
                'pajek': lambda g, p: nx.write_pajek(g, p)
            }
            
            for fmt, writer_func in formats.items():
                try:
                    file_path = os.path.join(output_dir, f"cleaned_graph.{fmt}")
                    writer_func(self.graph, file_path)
                    exported_files[fmt] = file_path
                    print(f"✅ Exported {fmt.upper()} format to {file_path}")
                except Exception as e:
                    print(f"⚠️ Could not export {fmt.upper()}: {e}")
            
            # CSV edge list with attributes
            try:
                edges_data = []
                for source, target, data in self.graph.edges(data=True):
                    edge_row = {'source': source, 'target': target}
                    edge_row.update(data)
                    edges_data.append(edge_row)
                
                if edges_data:
                    edges_df = pd.DataFrame(edges_data)
                    csv_path = os.path.join(output_dir, "cleaned_graph_edges.csv")
                    edges_df.to_csv(csv_path, index=False)
                    exported_files['csv_edges'] = csv_path
                    print(f"✅ Exported CSV edges to {csv_path}")
            except Exception as e:
                print(f"⚠️ Could not export CSV edges: {e}")
            
            # Node attributes CSV
            try:
                nodes_data = []
                for node, data in self.graph.nodes(data=True):
                    node_row = {'node_id': node}
                    node_row.update(data)
                    nodes_data.append(node_row)
                
                if nodes_data:
                    nodes_df = pd.DataFrame(nodes_data)
                    nodes_csv_path = os.path.join(output_dir, "cleaned_graph_nodes.csv")
                    nodes_df.to_csv(nodes_csv_path, index=False)
                    exported_files['csv_nodes'] = nodes_csv_path
                    print(f"✅ Exported CSV nodes to {nodes_csv_path}")
            except Exception as e:
                print(f"⚠️ Could not export CSV nodes: {e}")
            
            # JSON format
            try:
                json_data = nx.node_link_data(self.graph)
                json_path = os.path.join(output_dir, "cleaned_graph.json")
                with open(json_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
                exported_files['json'] = json_path
                print(f"✅ Exported JSON to {json_path}")
            except Exception as e:
                print(f"⚠️ Could not export JSON: {e}")
            
        except Exception as e:
            print(f"❌ Error during export: {e}")
        
        return exported_files