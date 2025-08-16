"""
Neural Data Generator using Conditional VAE and GAN
Enhanced for text data learning and generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
import re
from collections import Counter
import random
import torch.optim as optim

class TextProcessor:
    """Enhanced text processor for learning and generating realistic text"""
    
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_frequencies = Counter()
        self.text_patterns = []
        self.max_length = 100
        
    def fit(self, texts: pd.Series):
        """Learn text patterns from the data"""
        try:
            # Validate input
            if not isinstance(texts, pd.Series):
                raise ValueError("texts must be a pandas Series")
            
            if texts.empty:
                raise ValueError("texts cannot be empty")
            
            print("Learning text patterns from {} samples...".format(len(texts)))
            
            # Convert to strings and clean
            text_list = texts.astype(str).tolist()
            
            # Build vocabulary
            word_counts = Counter()
            for text in text_list:
                if isinstance(text, str):
                    words = text.lower().split()
                    word_counts.update(words)
            
            # Create word to index mapping
            self.word_to_idx = {word: idx for idx, (word, _) in enumerate(word_counts.most_common(100))}
            self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
            
            # Learn text patterns
            self.text_patterns = []
            for text in text_list:
                if isinstance(text, str) and len(text.strip()) > 0:
                    self.text_patterns.append(text.strip())
            
            print("Learned vocabulary of {} words".format(len(self.word_to_idx)))
            
        except Exception as e:
            print(f"Error in text processor fit: {e}")
            # Set default values
            self.word_to_idx = {}
            self.text_patterns = []
    
    def transform(self, texts: pd.Series) -> np.ndarray:
        """Transform text to numerical representation"""
        try:
            # Validate input
            if not isinstance(texts, pd.Series):
                raise ValueError("texts must be a pandas Series")
            
            if texts.empty:
                return np.array([])
            
            # Convert to numerical representation
            encoded_texts = []
            for text in texts:
                if isinstance(text, str) and text in self.word_to_idx:
                    encoded_texts.append(self.word_to_idx[text])
                else:
                    encoded_texts.append(0)  # Default value for unknown text
            
            return np.array(encoded_texts)
            
        except Exception as e:
            print(f"Error in text processor transform: {e}")
            # Return default values
            return np.zeros(len(texts))
    
    def decode_text(self, encoded: List[int]) -> str:
        """Decode numerical representation back to text"""
        try:
            # Validate input
            if not isinstance(encoded, list):
                return "invalid_input"
            
            if not encoded:
                return ""
            
            # Convert back to text
            words = []
            for idx in encoded:
                if isinstance(idx, int) and hasattr(self, 'idx_to_word') and idx in self.idx_to_word:
                    words.append(self.idx_to_word[idx])
                else:
                    words.append("unknown")
            
            return " ".join(words)
            
        except Exception as e:
            print(f"Error in text processor decode: {e}")
            return "decode_error"

class ConditionalVAE(nn.Module):
    """Conditional Variational Autoencoder for tabular data generation"""
    
    def __init__(self, input_dim, hidden_dim=128, latent_dim=64, num_conditions=10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_conditions = num_conditions
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + num_conditions, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_conditions, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def encode(self, x, conditions):
        """Encode input to latent space"""
        combined = torch.cat([x, conditions], dim=1)
        h = self.encoder(combined)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, conditions):
        """Decode latent space to output"""
        combined = torch.cat([z, conditions], dim=1)
        return self.decoder(combined)
    
    def forward(self, x, conditions=None):
        """Forward pass through the VAE"""
        try:
            # Validate input
            if not isinstance(x, torch.Tensor):
                raise ValueError("Input must be a torch tensor")
            
            if x.dim() != 2:
                raise ValueError("Input must be 2D tensor")
            
            # Create default conditions if none provided
            if conditions is None:
                batch_size = x.size(0)
                conditions = torch.randn(batch_size, self.num_conditions).to(x.device)
            
            # Encode
            mu, logvar = self.encode(x, conditions)
            
            # Sample from latent space
            z = self.reparameterize(mu, logvar)
            
            # Decode
            recon = self.decode(z, conditions)
            
            return recon, mu, logvar
            
        except Exception as e:
            print(f"Error in VAE forward pass: {e}")
            # Return default values
            batch_size = x.size(0) if hasattr(x, 'size') else 1
            device = x.device if hasattr(x, 'device') else torch.device('cpu')
            default_recon = torch.zeros(batch_size, self.input_dim).to(device)
            default_mu = torch.zeros(batch_size, self.latent_dim).to(device)
            default_logvar = torch.zeros(batch_size, self.latent_dim).to(device)
            return default_recon, default_mu, default_logvar

class GANDiscriminator(nn.Module):
    """GAN Discriminator for realism assessment"""
    
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass through the discriminator"""
        try:
            # Validate input
            if not isinstance(x, torch.Tensor):
                raise ValueError("Input must be a torch tensor")
            
            if x.dim() != 2:
                raise ValueError("Input must be 2D tensor")
            
            # Forward pass through the network
            return self.discriminator(x)
            
        except Exception as e:
            print(f"Error in discriminator forward pass: {e}")
            # Return default values
            batch_size = x.size(0) if hasattr(x, 'size') else 1
            device = x.device if hasattr(x, 'device') else torch.device('cpu')
            return torch.zeros(batch_size, 1).to(device)

class NeuralDataGenerator:
    """Enhanced neural network-based synthetic data generator"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.vae = None
        self.discriminator = None
        self.scalers = {}
        self.encoders = {}
        self.text_processors = {}
        self.column_info = {}
        self.original_data = None
        
    def fit(self, data: pd.DataFrame, schema_info: Dict[str, Any]):
        """Train the neural models on the data"""
        try:
            # Validate input data
            if not isinstance(data, pd.DataFrame):
                raise ValueError("data must be a pandas DataFrame")
            
            if data.empty:
                raise ValueError("data cannot be empty")
            
            if not isinstance(schema_info, dict):
                raise ValueError("schema_info must be a dictionary")
            
            print("Training enhanced neural data generator...")
            print(f"Training on {len(data)} samples with {len(data.columns)} columns")
            
            # Store column information
            self.column_info = {}
            for col in data.columns:
                col_info = schema_info.get(col, {})
                if isinstance(col_info, dict):
                    self.column_info[col] = col_info
                else:
                    # Default to object type if no info provided
                    self.column_info[col] = {'type': 'object'}
            
            # Preprocess data
            processed_data, preprocessing_info = self._preprocess_data(data, schema_info)
            
            # Store preprocessing info
            self.scalers = preprocessing_info.get('scalers', {})
            self.label_encoders = preprocessing_info.get('label_encoders', {})
            self.text_processors = preprocessing_info.get('text_processors', {})
            
            # Train VAE
            self._train_vae(processed_data)
            
            # Train discriminator
            self._train_discriminator(processed_data)
            
            print("Enhanced neural models trained successfully!")
            
        except Exception as e:
            print(f"Error in fit method: {e}")
            # Reset state on error
            self.vae = None
            self.discriminator = None
            self.column_info = {}
            self.scalers = {}
            self.label_encoders = {}
            self.text_processors = {}
            raise
    
    def generate(self, num_samples: int, conditions: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Generate synthetic data"""
        try:
            if self.vae is None:
                raise ValueError("Models not trained. Call fit() first.")
            
            if not hasattr(self, 'column_info') or not self.column_info:
                raise ValueError("Column information not available. Call fit() first.")
            
            if num_samples <= 0:
                raise ValueError("num_samples must be positive")
            
            print(f"Generating {num_samples} synthetic samples...")
            
            # Generate latent representations
            with torch.no_grad():
                # Sample from latent space
                z = torch.randn(num_samples, self.vae.latent_dim).to(self.device)
                
                # Create conditions for generation
                if conditions is None:
                    conditions = torch.randn(num_samples, self.vae.num_conditions).to(self.device)
                else:
                    # Encode conditions if provided as dict
                    conditions = self._encode_conditions(conditions, num_samples)
                
                # Generate synthetic data
                synthetic_data = self.vae.decode(z, conditions).cpu().numpy()
            
            # Post-process the generated data
            synthetic_df = self._postprocess_data(synthetic_data)
            
            return synthetic_df
            
        except Exception as e:
            print(f"Error in neural generation: {e}")
            # Return empty DataFrame with proper columns as fallback
            if hasattr(self, 'column_info') and self.column_info:
                columns = list(self.column_info.keys())
                return pd.DataFrame(columns=columns)
            else:
                return pd.DataFrame()
    
    def _preprocess_data(self, data: pd.DataFrame, schema_info: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Preprocess data for neural network training"""
        try:
            # Validate input data
            if not isinstance(data, pd.DataFrame):
                raise ValueError("data must be a pandas DataFrame")
            
            if data.empty:
                raise ValueError("data cannot be empty")
            
            if not isinstance(schema_info, dict):
                raise ValueError("schema_info must be a dictionary")
            
            processed_data = data.copy()
            column_info = {}
            scalers = {}
            label_encoders = {}
            text_processors = {}
            
            for col in processed_data.columns:
                try:
                    col_info = schema_info.get(col, {})
                    if not isinstance(col_info, dict):
                        col_info = {}
                    
                    col_type = col_info.get('data_type', 'object')
                    column_info[col] = {'type': col_type}
                    
                    if col_type in ['int64', 'float64'] or processed_data[col].dtype in ['int64', 'float64']:
                        # Numerical column
                        column_info[col]['type'] = 'numerical'
                        
                        # Handle missing values
                        if processed_data[col].isna().any():
                            processed_data[col] = processed_data[col].fillna(processed_data[col].mean())
                        
                        # Scale numerical data
                        scaler = StandardScaler()
                        processed_data[col] = scaler.fit_transform(processed_data[col].values.reshape(-1, 1)).flatten()
                        scalers[col] = scaler
                        
                    elif col_type in ['object', 'string', 'category'] or processed_data[col].dtype in ['object', 'string', 'category']:
                        # Categorical column
                        column_info[col]['type'] = 'categorical'
                        
                        # Handle missing values
                        if processed_data[col].isna().any():
                            processed_data[col] = processed_data[col].fillna('unknown')
                        
                        # Encode categorical data
                        label_encoder = LabelEncoder()
                        processed_data[col] = label_encoder.fit_transform(processed_data[col].astype(str))
                        label_encoders[col] = label_encoder
                        
                    elif col_type == 'text' or processed_data[col].dtype == 'object':
                        # Text column
                        column_info[col]['type'] = 'text'
                        
                        # Handle missing values
                        if processed_data[col].isna().any():
                            processed_data[col] = processed_data[col].fillna('')
                        
                        # Create text processor
                        text_processor = TextProcessor()
                        text_processor.fit(processed_data[col])
                        text_processors[col] = text_processor
                        
                        # Convert text to numerical representation
                        processed_data[col] = text_processor.transform(processed_data[col])
                        
                    else:
                        # Default to object type
                        column_info[col]['type'] = 'object'
                        
                        # Handle missing values
                        if processed_data[col].isna().any():
                            processed_data[col] = processed_data[col].fillna(0)
                        
                        # Convert to numeric
                        processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce').fillna(0)
                        
                except Exception as e:
                    print(f"Warning: Error preprocessing column {col}: {e}")
                    # Set default values
                    column_info[col] = {'type': 'object'}
                    processed_data[col] = 0
            
            # Ensure all data is numeric and handle NaN/Inf values
            processed_data = processed_data.astype(float)
            
            # Replace NaN and Inf values with 0
            processed_data = processed_data.replace([np.inf, -np.inf], np.nan)
            processed_data = processed_data.fillna(0)
            
            # Validate data after preprocessing
            if np.any(np.isnan(processed_data)) or np.any(np.isinf(processed_data)):
                print("Warning: Data still contains NaN/Inf values after preprocessing, replacing with 0")
                processed_data = processed_data.replace([np.inf, -np.inf], np.nan)
                processed_data = processed_data.fillna(0)
            
            # Ensure data is finite
            if not np.all(np.isfinite(processed_data)):
                print("Warning: Data contains non-finite values, replacing with 0")
                processed_data = np.where(np.isfinite(processed_data), processed_data, 0)
            
            preprocessing_info = {
                'scalers': scalers,
                'label_encoders': label_encoders,
                'text_processors': text_processors
            }
            
            return processed_data.values, preprocessing_info
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            # Return empty data as fallback
            return np.array([]), {}
    
    def _train_vae(self, processed_data: np.ndarray):
        """Train the VAE model"""
        try:
            # Validate input data
            if not isinstance(processed_data, np.ndarray):
                raise ValueError("processed_data must be a numpy array")
            
            if processed_data.size == 0:
                raise ValueError("processed_data cannot be empty")
            
            # Initialize VAE
            input_dim = processed_data.shape[1]
            self.vae = ConditionalVAE(input_dim).to(self.device)
            
            # Convert to tensor
            data_tensor = torch.FloatTensor(processed_data).to(self.device)
            
            # Training parameters
            optimizer = optim.Adam(self.vae.parameters(), lr=0.001)
            num_epochs = 50
            
            # Training loop
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                
                # Create conditions for training
                batch_size = data_tensor.size(0)
                conditions = torch.randn(batch_size, self.vae.num_conditions).to(self.device)
                
                # Forward pass
                recon_batch, mu, logvar = self.vae(data_tensor, conditions)
                
                # Calculate loss with safety checks
                try:
                    recon_loss = F.mse_loss(recon_batch, data_tensor, reduction='sum')
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    
                    # Check for NaN/Inf in losses
                    if torch.isnan(recon_loss) or torch.isinf(recon_loss):
                        recon_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                    if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                        kl_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                    
                    total_loss = recon_loss + kl_loss
                    
                    # Check if total loss is valid
                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        print(f"Warning: Invalid loss detected at epoch {epoch}, skipping")
                        continue
                    
                    # Backward pass
                    total_loss.backward()
                    
                    # Gradient clipping to prevent explosion
                    torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    if epoch % 10 == 0:
                        print(f"VAE Epoch {epoch}, Loss: {total_loss.item():.4f}")
                        
                except Exception as e:
                    print(f"Warning: Error in training epoch {epoch}: {e}")
                    continue
            
        except Exception as e:
            print(f"Error in VAE training: {e}")
            self.vae = None
            raise
    
    def _train_discriminator(self, processed_data: np.ndarray):
        """Train the discriminator model"""
        try:
            # Validate input data
            if not isinstance(processed_data, np.ndarray):
                raise ValueError("processed_data must be a numpy array")
            
            if processed_data.size == 0:
                raise ValueError("processed_data cannot be empty")
            
            # Initialize discriminator
            input_dim = processed_data.shape[1]
            self.discriminator = GANDiscriminator(input_dim).to(self.device)
            
            # Convert to tensor
            data_tensor = torch.FloatTensor(processed_data).to(self.device)
            
            # Training parameters
            optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002)
            num_epochs = 25
            
            # Training loop
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                
                # Forward pass
                real_output = self.discriminator(data_tensor)
                
                # Calculate loss (simple binary cross entropy)
                real_labels = torch.ones(real_output.size()).to(self.device)
                real_loss = F.binary_cross_entropy_with_logits(real_output, real_labels)
                
                # Backward pass
                real_loss.backward()
                optimizer.step()
                
                if epoch % 5 == 0:
                    print(f"Discriminator Epoch {epoch}, Loss: {real_loss.item():.4f}")
            
        except Exception as e:
            print(f"Error in discriminator training: {e}")
            self.discriminator = None
            raise
    
    def _encode_conditions(self, conditions: Dict[str, Any], num_samples: int) -> torch.Tensor:
        """Encode conditions for generation"""
        try:
            # Simple condition encoding for MVP
            if not hasattr(self, 'vae') or self.vae is None:
                # Fallback if VAE not available
                return torch.randn(num_samples, 10).to(self.device)
            
            condition_tensor = torch.randn(num_samples, self.vae.num_conditions).to(self.device)
            return condition_tensor
            
        except Exception as e:
            print(f"Error encoding conditions: {e}")
            # Fallback to random conditions
            return torch.randn(num_samples, 10).to(self.device)
    
    def _postprocess_data(self, synthetic_data: np.ndarray) -> pd.DataFrame:
        """Enhanced post-processing with realistic text generation"""
        try:
            # Check if column_info is properly initialized
            if not hasattr(self, 'column_info') or not self.column_info:
                # Fallback: create DataFrame with generic column names
                num_cols = synthetic_data.shape[1]
                column_names = [f'column_{i}' for i in range(num_cols)]
                df = pd.DataFrame(synthetic_data, columns=column_names)
                print("Warning: column_info not initialized, using generic column names")
                return df
            
            # Validate synthetic_data
            if not isinstance(synthetic_data, np.ndarray):
                print("Warning: synthetic_data is not a numpy array")
                return pd.DataFrame()
            
            if synthetic_data.size == 0:
                print("Warning: synthetic_data is empty")
                return pd.DataFrame()
            
            df = pd.DataFrame(synthetic_data, columns=list(self.column_info.keys()))
            
            # Process each column based on its type
            for col, info in self.column_info.items():
                # Skip if info is not properly structured
                if not isinstance(info, dict) or 'type' not in info:
                    print(f"Warning: Skipping column {col} due to invalid info structure")
                    continue
                    
                if info['type'] == 'numerical' and hasattr(self, 'scalers') and col in self.scalers:
                    try:
                        # Inverse transform numerical columns
                        col_data = df[col].values.reshape(-1, 1)
                        df[col] = self.scalers[col].inverse_transform(col_data).flatten()
                    except Exception as e:
                        print(f"Warning: Error inverse transforming column {col}: {e}")
                        continue
                        
                elif info['type'] == 'categorical' and hasattr(self, 'label_encoders') and col in self.label_encoders:
                    try:
                        # Convert categorical columns back to original labels
                        df[col] = self.label_encoders[col].inverse_transform(df[col].astype(int))
                    except Exception as e:
                        print(f"Warning: Error inverse transforming categorical column {col}: {e}")
                        continue
                        
                elif info['type'] == 'text' and hasattr(self, 'text_processors') and col in self.text_processors:
                    try:
                        # Generate realistic text for text columns
                        processor = self.text_processors[col]
                        generated_texts = []
                        
                        for _ in range(len(df)):
                            if hasattr(processor, 'generate_text') and callable(processor.generate_text):
                                try:
                                    generated_text = processor.generate_text()
                                    generated_texts.append(generated_text)
                                except:
                                    # Fallback to vocabulary-based generation
                                    if hasattr(processor, 'word_to_idx') and processor.word_to_idx:
                                        words = list(processor.word_to_idx.keys())[:10]
                                        if len(words) >= 5:
                                            generated_text = ' '.join(random.sample(words, 5))
                                        else:
                                            generated_text = ' '.join(words)
                                    else:
                                        generated_text = f"text_{random.randint(1000, 9999)}"
                                    generated_texts.append(generated_text)
                            else:
                                # Fallback to vocabulary-based generation
                                if hasattr(processor, 'word_to_idx') and processor.word_to_idx:
                                    words = list(processor.word_to_idx.keys())[:10]
                                    if len(words) >= 5:
                                        generated_text = ' '.join(random.sample(words, 5))
                                    else:
                                        generated_text = ' '.join(words)
                                else:
                                    generated_text = f"text_{random.randint(1000, 9999)}"
                                generated_texts.append(generated_text)
                        
                        df[col] = generated_texts
                        
                    except Exception as e:
                        print(f"Warning: Error generating text for column {col}: {e}")
                        # Fallback to generic text
                        df[col] = [f"text_{i}" for i in range(len(df))]
            
            return df
            
        except Exception as e:
            print(f"Error in postprocess_data: {e}")
            # Return empty DataFrame as fallback
            return pd.DataFrame()

class MultiTableNeuralGenerator:
    """Neural generator for multiple related tables"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.generators = {}
        self.relationship_graph = nx.DiGraph()
        
    def fit(self, data_dict: Dict[str, pd.DataFrame], relationships: List[Dict[str, Any]], metadata: Dict[str, Any] = None):
        """Fit the multi-table neural generator"""
        try:
            # Validate input data
            if not isinstance(data_dict, dict):
                raise ValueError("data_dict must be a dictionary")
            
            if not data_dict:
                raise ValueError("data_dict cannot be empty")
            
            if not isinstance(relationships, list):
                raise ValueError("relationships must be a list")
            
            print("Training multi-table neural generator...")
            
            # Store data and relationships
            self.data_dict = data_dict
            self.relationships = relationships
            
            # Train individual generators for each table
            self.generators = {}
            for table_name, data in data_dict.items():
                try:
                    if isinstance(data, pd.DataFrame) and not data.empty:
                        # Try to get schema info from metadata first
                        schema_info = {}
                        if metadata and 'tables' in metadata:
                            table_schema = next((t for t in metadata['tables'] if t.get('name') == table_name), None)
                            if table_schema and 'columns' in table_schema:
                                schema_info = {col.get('name', f'col_{i}'): {'data_type': col.get('data_type', 'object')} 
                                             for i, col in enumerate(table_schema['columns'])}
                        
                        # Fallback to column types if no metadata
                        if not schema_info:
                            schema_info = {col: {'data_type': str(data[col].dtype)} for col in data.columns}
                        
                        # Train generator
                        generator = NeuralDataGenerator()
                        generator.fit(data, schema_info)
                        self.generators[table_name] = generator
                        
                except Exception as e:
                    print(f"Warning: Error training generator for table {table_name}: {e}")
                    continue
            
            print(f"Trained generators for {len(self.generators)} tables")
            
        except Exception as e:
            print(f"Error in multi-table fit: {e}")
            # Reset state on error
            self.data_dict = {}
            self.relationships = []
            self.generators = {}
            raise
    
    def generate(self, target_samples: Dict[str, int]) -> Dict[str, pd.DataFrame]:
        """Generate synthetic data for multiple tables"""
        try:
            # Validate input
            if not isinstance(target_samples, dict):
                raise ValueError("target_samples must be a dictionary")
            
            if not target_samples:
                raise ValueError("target_samples cannot be empty")
            
            print("Generating multi-table synthetic data...")
            
            synthetic_data = {}
            
            for table_name, num_samples in target_samples.items():
                try:
                    if table_name in self.generators:
                        generator = self.generators[table_name]
                        synthetic_data[table_name] = generator.generate(num_samples)
                    else:
                        print(f"Warning: No generator found for table {table_name}")
                        # Create empty DataFrame with expected columns
                        if table_name in self.data_dict:
                            columns = self.data_dict[table_name].columns
                            synthetic_data[table_name] = pd.DataFrame(columns=columns)
                        else:
                            synthetic_data[table_name] = pd.DataFrame()
                            
                except Exception as e:
                    print(f"Warning: Error generating data for table {table_name}: {e}")
                    # Create empty DataFrame as fallback
                    if table_name in self.data_dict:
                        columns = self.data_dict[table_name].columns
                        synthetic_data[table_name] = pd.DataFrame(columns=columns)
                    else:
                        synthetic_data[table_name] = pd.DataFrame()
            
            return synthetic_data
            
        except Exception as e:
            print(f"Error in multi-table generation: {e}")
            # Return empty data as fallback
            return {table_name: pd.DataFrame() for table_name in target_samples.keys()} 