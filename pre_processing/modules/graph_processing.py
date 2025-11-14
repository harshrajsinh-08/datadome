import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
import warnings
warnings.filterwarnings('ignore')

class GraphDatasetCleaner:
    def __init__(self):
        self.graph = None
        self.original_graph = None
        self.cleaning_stats = {
            'original_nodes': 0,
            'original_edges': 0,
            'isolated_nodes_removed': 0,
            'duplicate_edges_removed': 0,
            'self_loops_removed': 0,
            'low_degree_nodes_removed': 0,
            'disconnected_components_removed': 0,
            'final_nodes': 0,
            'final_edges': 0
        }
        
    def load_graph_from_edgelist(self, file_path, directed=False, weighted=False):
        """Load graph from edge list file"""
        try:
            print(f"Loading graph from {file_path}...")
            
            # Read edge list
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                # Assume first two columns are source and target
                if len(df.columns) >= 2:
                    edges = [(row[0], row[1]) for _, row in df.iterrows()]
                    if weighted and len(df.columns) >= 3:
                        # Third column is weight
                        edges = [(row[0], row[1], {'weight': row[2]}) for _, row in df.iterrows()]
                else:
                    raise ValueError("CSV must have at least 2 columns (source, target)")
            else:
                # Plain text edge list
                edges = []
                with open(file_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            if weighted and len(parts) >= 3:
                                edges.append((parts[0], parts[1], {'weight': float(parts[2])}))
                            else:
                                edges.append((parts[0], parts[1]))
            
            # Create graph
            if directed:
                self.graph = nx.DiGraph()
            else:
                self.graph = nx.Graph()
            
            self.graph.add_edges_from(edges)
            self.original_graph = self.graph.copy()
            
            # Update stats
            self.cleaning_stats['original_nodes'] = self.graph.number_of_nodes()
            self.cleaning_stats['original_edges'] = self.graph.number_of_edges()
            
            print(f"✅ Graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            return True
            
        except Exception as e:
            print(f"❌ Error loading graph: {e}")
            return False
    
    def load_graph_from_gml(self, file_path):
        """Load graph from GML file"""
        try:
            self.graph = nx.read_gml(file_path)
            self.original_graph = self.graph.copy()
            
            self.cleaning_stats['original_nodes'] = self.graph.number_of_nodes()
            self.cleaning_stats['original_edges'] = self.graph.number_of_edges()
            
            print(f"✅ GML graph loaded: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
            return True
        except Exception as e:
            print(f"❌ Error loading GML: {e}")
            return False
    
    def analyze_graph_quality(self):
        """Analyze graph structure and quality metrics"""
        if self.graph is None:
            return None
        
        analysis = {
            'basic_stats': {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges(),
                'density': nx.density(self.graph),
                'is_directed': self.graph.is_directed()
            },
            'connectivity': {
                'is_connected': nx.is_connected(self.graph) if not self.graph.is_directed() else nx.is_weakly_connected(self.graph),
                'num_components': nx.number_connected_components(self.graph) if not self.graph.is_directed() else nx.number_weakly_connected_components(self.graph),
                'largest_component_size': len(max(nx.connected_components(self.graph) if not self.graph.is_directed() else nx.weakly_connected_components(self.graph), key=len))
            },
            'degree_stats': {
                'avg_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0,
                'max_degree': max(dict(self.graph.degree()).values()) if self.graph.number_of_nodes() > 0 else 0,
                'min_degree': min(dict(self.graph.degree()).values()) if self.graph.number_of_nodes() > 0 else 0
            },
            'quality_issues': {
                'isolated_nodes': len([n for n in self.graph.nodes() if self.graph.degree(n) == 0]),
                'self_loops': nx.number_of_selfloops(self.graph),
                'duplicate_edges': 0  # Will be calculated
            }
        }
        
        # Calculate duplicate edges for undirected graphs
        if not self.graph.is_directed():
            edge_set = set()
            duplicates = 0
            for edge in self.graph.edges():
                sorted_edge = tuple(sorted(edge))
                if sorted_edge in edge_set:
                    duplicates += 1
                else:
                    edge_set.add(sorted_edge)
            analysis['quality_issues']['duplicate_edges'] = duplicates
        
        return analysis
    
    def clean_graph(self, 
                   remove_isolated_nodes=True,
                   remove_self_loops=True,
                   remove_duplicate_edges=True,
                   min_degree_threshold=1,
                   keep_largest_component=True,
                   min_component_size=5):
        """Clean the graph based on specified criteria"""
        
        if self.graph is None:
            print("❌ No graph loaded")
            return False
        
        print("🧹 Starting graph cleaning...")
        
        # 1. Remove self loops
        if remove_self_loops:
            self_loops = list(nx.selfloop_edges(self.graph))
            self.graph.remove_edges_from(self_loops)
            self.cleaning_stats['self_loops_removed'] = len(self_loops)
            print(f"Removed {len(self_loops)} self loops")
        
        # 2. Remove duplicate edges (for MultiGraph)
        if remove_duplicate_edges and hasattr(self.graph, 'remove_edges_from'):
            # Convert to simple graph to remove duplicates
            if isinstance(self.graph, (nx.MultiGraph, nx.MultiDiGraph)):
                original_edges = self.graph.number_of_edges()
                self.graph = nx.Graph(self.graph) if not self.graph.is_directed() else nx.DiGraph(self.graph)
                duplicates_removed = original_edges - self.graph.number_of_edges()
                self.cleaning_stats['duplicate_edges_removed'] = duplicates_removed
                print(f"Removed {duplicates_removed} duplicate edges")
        
        # 3. Remove low degree nodes (be more careful with directed graphs)
        if min_degree_threshold > 0:
            if self.graph.is_directed():
                # For directed graphs, only remove nodes with BOTH low in-degree AND low out-degree
                # This preserves important directional patterns
                low_degree_nodes = [n for n in self.graph.nodes() 
                                  if self.graph.in_degree(n) < min_degree_threshold 
                                  and self.graph.out_degree(n) < min_degree_threshold]
            else:
                # For undirected graphs, use total degree
                low_degree_nodes = [n for n in self.graph.nodes() 
                                  if self.graph.degree(n) < min_degree_threshold]
            
            self.graph.remove_nodes_from(low_degree_nodes)
            self.cleaning_stats['low_degree_nodes_removed'] = len(low_degree_nodes)
            print(f"Removed {len(low_degree_nodes)} nodes with degree < {min_degree_threshold}")
        
        # 4. Remove isolated nodes
        if remove_isolated_nodes:
            isolated_nodes = [n for n in self.graph.nodes() if self.graph.degree(n) == 0]
            self.graph.remove_nodes_from(isolated_nodes)
            self.cleaning_stats['isolated_nodes_removed'] = len(isolated_nodes)
            print(f"Removed {len(isolated_nodes)} isolated nodes")
        
        # 5. Keep only largest component or remove small components
        if keep_largest_component:
            if self.graph.is_directed():
                # For directed graphs, use weakly connected components
                # But be more conservative - only remove very small components
                components = list(nx.weakly_connected_components(self.graph))
                # Increase minimum component size for directed graphs to preserve structure
                effective_min_size = max(min_component_size, 10) if self.graph.is_directed() else min_component_size
            else:
                components = list(nx.connected_components(self.graph))
                effective_min_size = min_component_size
            
            if len(components) > 1:
                # Keep largest component and larger secondary components
                largest_component = max(components, key=len)
                nodes_to_remove = []
                for component in components:
                    if component != largest_component:
                        if len(component) < effective_min_size:
                            nodes_to_remove.extend(component)
                
                self.graph.remove_nodes_from(nodes_to_remove)
                self.cleaning_stats['disconnected_components_removed'] = len(nodes_to_remove)
                print(f"Removed {len(nodes_to_remove)} nodes from small components (min size: {effective_min_size})")
        
        # Update final stats
        self.cleaning_stats['final_nodes'] = self.graph.number_of_nodes()
        self.cleaning_stats['final_edges'] = self.graph.number_of_edges()
        
        print(f"✅ Graph cleaning completed:")
        print(f"  Nodes: {self.cleaning_stats['original_nodes']} → {self.cleaning_stats['final_nodes']}")
        print(f"  Edges: {self.cleaning_stats['original_edges']} → {self.cleaning_stats['final_edges']}")
        
        return True
    
    def export_cleaned_graph(self, output_path, format='edgelist'):
        """Export cleaned graph to file"""
        if self.graph is None:
            return False
        
        try:
            if format == 'edgelist':
                nx.write_edgelist(self.graph, output_path)
            elif format == 'gml':
                nx.write_gml(self.graph, output_path)
            elif format == 'graphml':
                nx.write_graphml(self.graph, output_path)
            elif format == 'csv':
                # Export as CSV edge list
                edges_df = pd.DataFrame(self.graph.edges(), columns=['source', 'target'])
                edges_df.to_csv(output_path, index=False)
            
            print(f"✅ Cleaned graph exported to {output_path}")
            return True
        except Exception as e:
            print(f"❌ Error exporting graph: {e}")
            return False
    
    def calculate_graph_metrics(self, graph=None):
        """Calculate comprehensive graph metrics"""
        if graph is None:
            graph = self.graph
        
        if graph is None or graph.number_of_nodes() == 0:
            return None
        
        metrics = {}
        
        try:
            # Basic metrics
            metrics['nodes'] = graph.number_of_nodes()
            metrics['edges'] = graph.number_of_edges()
            metrics['density'] = nx.density(graph)
            
            # Degree metrics
            degrees = dict(graph.degree())
            metrics['avg_degree'] = np.mean(list(degrees.values()))
            metrics['degree_std'] = np.std(list(degrees.values()))
            metrics['max_degree'] = max(degrees.values()) if degrees else 0
            
            # Connectivity metrics
            if graph.is_directed():
                metrics['is_connected'] = nx.is_weakly_connected(graph)
                metrics['num_components'] = nx.number_weakly_connected_components(graph)
            else:
                metrics['is_connected'] = nx.is_connected(graph)
                metrics['num_components'] = nx.number_connected_components(graph)
            
            # Centrality metrics (for smaller graphs)
            if graph.number_of_nodes() < 1000:
                try:
                    betweenness = nx.betweenness_centrality(graph)
                    metrics['avg_betweenness'] = np.mean(list(betweenness.values()))
                    
                    closeness = nx.closeness_centrality(graph)
                    metrics['avg_closeness'] = np.mean(list(closeness.values()))
                    
                    eigenvector = nx.eigenvector_centrality(graph, max_iter=1000)
                    metrics['avg_eigenvector'] = np.mean(list(eigenvector.values()))
                except:
                    metrics['avg_betweenness'] = 0
                    metrics['avg_closeness'] = 0
                    metrics['avg_eigenvector'] = 0
            else:
                metrics['avg_betweenness'] = 0
                metrics['avg_closeness'] = 0
                metrics['avg_eigenvector'] = 0
            
            # Clustering
            if not graph.is_directed():
                metrics['avg_clustering'] = nx.average_clustering(graph)
            else:
                metrics['avg_clustering'] = 0
            
            # Path metrics (for connected graphs)
            if metrics['is_connected'] and graph.number_of_nodes() < 500:
                try:
                    metrics['avg_shortest_path'] = nx.average_shortest_path_length(graph)
                    metrics['diameter'] = nx.diameter(graph)
                except:
                    metrics['avg_shortest_path'] = 0
                    metrics['diameter'] = 0
            else:
                metrics['avg_shortest_path'] = 0
                metrics['diameter'] = 0
                
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return None
        
        return metrics