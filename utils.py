import pandas as pd
import numpy as np

def consolidate_metrics(results):
    metrics_df = pd.DataFrame()
    for m in range(len(results)):
        metrics_df = pd.concat([metrics_df, pd.DataFrame([results[m]])], axis=0) 
    return metrics_df