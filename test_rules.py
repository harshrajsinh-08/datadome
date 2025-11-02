#!/usr/bin/env python3

import pandas as pd
import sys
import os

# Add the project root to Python path
sys.path.append('.')

from pre_processing.modules.rule_based_cleaning import rule_based_cleaning

def test_rule_cleaning():
    """Test rule-based cleaning"""
    
    # Read the current clean data
    df = pd.read_csv('output/clean_user_data.csv')
    print("Original data shape:", df.shape)
    print("Original data columns:", df.columns.