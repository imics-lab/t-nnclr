import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from pathlib import Path

def load_and_preprocess_har_data(data_dir='./data', logger=None):
    """
    Load and preprocess HAR dataset.
    
    Args:
        data_dir: Directory containing the data files
        logger: Optional logger instance
    
    Returns:
        tuple: (X_unlabeled, X_labeled, y, subjects)
            - X_unlabeled: Unlabeled data array
            - X_labeled: Labeled data array
            - y: Labels array
            - subjects: Subject IDs array
    """
    try:
        data_dir = Path(data_dir)
        
        # Load data
        # X_unlabeled = np.load(data_dir / 'X_unlabeled.npy')
        X_unlabeled = np.load(data_dir / 'X.npy')
        X_labeled = np.load(data_dir / 'X.npy')
        y = np.load(data_dir / 'y.npy').squeeze()
        subjects = np.load(data_dir / 'sub.npy').squeeze()
        
        # Use only first 4 channels of the data
        X_unlabeled = X_unlabeled[:, :, :4]
        X_labeled = X_labeled[:, :, :4]
        
        if logger:
            logger.info(f"Loaded data shapes:")
            logger.info(f"X_unlabeled: {X_unlabeled.shape}")
            logger.info(f"X_labeled: {X_labeled.shape}")
            logger.info(f"y: {y.shape}")
            logger.info(f"subjects: {subjects.shape}")
            logger.info(f"Number of unique subjects: {len(np.unique(subjects))}")
            logger.info(f"Number of unique classes: {len(np.unique(y))}")
        
        # Convert labels to integers
        le = LabelEncoder()
        y = le.fit_transform(y)
        if logger:
            logger.info("Labels encoded successfully")
        
        # Standardize the data
        scaler = StandardScaler()
        
        # Reshape for standardization
        orig_shape_unlabeled = X_unlabeled.shape
        orig_shape_labeled = X_labeled.shape
        
        # Combine all data for computing statistics
        combined = np.vstack([
            X_unlabeled.reshape(-1, X_unlabeled.shape[-1]),
            X_labeled.reshape(-1, X_labeled.shape[-1])
        ])
        
        # Fit on combined data
        scaler.fit(combined)
        
        # Transform separately and reshape back
        X_unlabeled = scaler.transform(
            X_unlabeled.reshape(-1, X_unlabeled.shape[-1])
        ).reshape(orig_shape_unlabeled)
        
        X_labeled = scaler.transform(
            X_labeled.reshape(-1, X_labeled.shape[-1])
        ).reshape(orig_shape_labeled)
        
        if logger:
            logger.info("Data standardization completed")
        
        return X_unlabeled, X_labeled, y, subjects
        
    except Exception as e:
        if logger:
            logger.error(f"Error loading data: {str(e)}")
        raise

if __name__ == "__main__":
    # Test data loading
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    data_dir = Path.cwd() / 'data'
    X_unlabeled, X_labeled, y, subjects = load_and_preprocess_har_data(data_dir, logger)
    
    print("\nData loading test completed successfully!")
    print(f"Unlabeled data shape: {X_unlabeled.shape}")
    print(f"Labeled data shape: {X_labeled.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Subjects shape: {subjects.shape}")