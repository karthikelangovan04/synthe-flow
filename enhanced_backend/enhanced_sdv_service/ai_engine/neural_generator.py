"""
Neural Data Generator using Conditional VAE and GAN
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
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_conditions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
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
    
    def forward(self, x, conditions):
        """Forward pass"""
        mu, log_var = self.encode(x, conditions)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, conditions), mu, log_var

class GANDiscriminator(nn.Module):
    """GAN Discriminator for realism assessment"""
    
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.discriminator(x)

class NeuralDataGenerator:
    """Neural network-based synthetic data generator"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.vae = None
        self.discriminator = None
        self.scalers = {}
        self.encoders = {}
        self.column_info = {}
        
    def fit(self, data: pd.DataFrame, schema_info: Dict[str, Any]):
        """Train the neural models on the data"""
        print("Training neural data generator...")
        
        # Preprocess data
        processed_data, column_info = self._preprocess_data(data, schema_info)
        self.column_info = column_info
        
        # Initialize models
        input_dim = processed_data.shape[1]
        self.vae = ConditionalVAE(input_dim).to(self.device)
        self.discriminator = GANDiscriminator(input_dim).to(self.device)
        
        # Train models
        self._train_vae(processed_data)
        self._train_discriminator(processed_data)
        
        print("Neural models trained successfully!")
    
    def generate(self, num_samples: int, conditions: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Generate synthetic data"""
        if self.vae is None:
            raise ValueError("Models not trained. Call fit() first.")
        
        print(f"Generating {num_samples} synthetic samples...")
        
        # Generate latent samples
        z = torch.randn(num_samples, self.vae.latent_dim).to(self.device)
        
        # Create conditions
        if conditions is None:
            conditions = torch.randn(num_samples, self.vae.num_conditions).to(self.device)
        else:
            conditions = self._encode_conditions(conditions, num_samples)
        
        # Generate synthetic data
        with torch.no_grad():
            synthetic_data = self.vae.decode(z, conditions)
        
        # Post-process data
        synthetic_df = self._postprocess_data(synthetic_data.cpu().numpy())
        
        return synthetic_df
    
    def _preprocess_data(self, data: pd.DataFrame, schema_info: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Preprocess data for neural training"""
        processed_data = data.copy()
        column_info = {}
        
        for col in data.columns:
            col_info = schema_info.get(col, {})
            data_type = col_info.get('data_type', 'unknown')
            
            if data_type in ['integer', 'decimal', 'float']:
                # Scale numerical data
                scaler = StandardScaler()
                processed_data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1)).flatten()
                self.scalers[col] = scaler
                column_info[col] = {'type': 'numerical', 'scaler': scaler}
                
            elif data_type in ['varchar', 'text']:
                # Encode categorical data
                if data[col].nunique() < 100:  # Only encode if not too many unique values
                    encoder = LabelEncoder()
                    processed_data[col] = encoder.fit_transform(data[col].astype(str))
                    self.encoders[col] = encoder
                    column_info[col] = {'type': 'categorical', 'encoder': encoder}
                else:
                    # For high-cardinality text, use simple encoding
                    processed_data[col] = pd.factorize(data[col])[0]
                    column_info[col] = {'type': 'text', 'factorized': True}
                    
            elif data_type == 'boolean':
                # Convert boolean to integer
                processed_data[col] = data[col].astype(int)
                column_info[col] = {'type': 'boolean'}
                
            elif data_type == 'timestamp':
                # Convert timestamp to numerical features
                processed_data[col] = pd.to_datetime(data[col]).astype(np.int64) // 10**9
                scaler = StandardScaler()
                processed_data[col] = scaler.fit_transform(processed_data[col].values.reshape(-1, 1)).flatten()
                self.scalers[col] = scaler
                column_info[col] = {'type': 'timestamp', 'scaler': scaler}
            
            else:
                # Default to numerical
                processed_data[col] = pd.factorize(data[col])[0]
                column_info[col] = {'type': 'unknown', 'factorized': True}
        
        return processed_data.values, column_info
    
    def _train_vae(self, data: np.ndarray, epochs=100):
        """Train the VAE"""
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
        
        # Create dummy conditions for training
        conditions = torch.randn(len(data), self.vae.num_conditions).to(self.device)
        data_tensor = torch.FloatTensor(data).to(self.device)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            recon, mu, log_var = self.vae(data_tensor, conditions)
            
            # Loss calculation
            recon_loss = F.mse_loss(recon, data_tensor)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            total_loss = recon_loss + 0.1 * kl_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"VAE Epoch {epoch}, Loss: {total_loss.item():.4f}")
    
    def _train_discriminator(self, data: np.ndarray, epochs=50):
        """Train the discriminator"""
        optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3)
        criterion = nn.BCELoss()
        
        data_tensor = torch.FloatTensor(data).to(self.device)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Real data
            real_labels = torch.ones(len(data), 1).to(self.device)
            real_output = self.discriminator(data_tensor)
            real_loss = criterion(real_output, real_labels)
            
            # Fake data
            z = torch.randn(len(data), self.vae.latent_dim).to(self.device)
            conditions = torch.randn(len(data), self.vae.num_conditions).to(self.device)
            fake_data = self.vae.decode(z, conditions)
            fake_labels = torch.zeros(len(data), 1).to(self.device)
            fake_output = self.discriminator(fake_data.detach())
            fake_loss = criterion(fake_output, fake_labels)
            
            # Total loss
            total_loss = real_loss + fake_loss
            total_loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Discriminator Epoch {epoch}, Loss: {total_loss.item():.4f}")
    
    def _encode_conditions(self, conditions: Dict[str, Any], num_samples: int) -> torch.Tensor:
        """Encode conditions for generation"""
        # Simple condition encoding for MVP
        condition_tensor = torch.randn(num_samples, self.vae.num_conditions).to(self.device)
        return condition_tensor
    
    def _postprocess_data(self, synthetic_data: np.ndarray) -> pd.DataFrame:
        """Post-process synthetic data back to original format"""
        df = pd.DataFrame(synthetic_data, columns=list(self.column_info.keys()))
        
        for col, info in self.column_info.items():
            if info['type'] == 'numerical' and col in self.scalers:
                # Inverse transform numerical data
                df[col] = self.scalers[col].inverse_transform(df[col].values.reshape(-1, 1)).flatten()
                df[col] = df[col].round().astype(int)  # Round to integers
                
            elif info['type'] == 'categorical' and col in self.encoders:
                # Inverse transform categorical data
                df[col] = self.encoders[col].inverse_transform(df[col].round().astype(int))
                
            elif info['type'] == 'boolean':
                # Convert back to boolean
                df[col] = (df[col] > 0.5).astype(int)
                
            elif info['type'] == 'timestamp' and col in self.scalers:
                # Convert back to timestamp
                df[col] = self.scalers[col].inverse_transform(df[col].values.reshape(-1, 1)).flatten()
                df[col] = pd.to_datetime(df[col], unit='s')
                
            elif info.get('factorized', False):
                # For factorized data, just round to integers
                df[col] = df[col].round().astype(int)
        
        return df

class MultiTableNeuralGenerator:
    """Neural generator for multiple related tables"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.generators = {}
        self.relationship_graph = nx.DiGraph()
        
    def fit(self, tables_data: Dict[str, pd.DataFrame], relationships: List[Dict[str, Any]]):
        """Train generators for multiple tables"""
        print("Training multi-table neural generators...")
        
        # Build relationship graph
        self._build_relationship_graph(relationships)
        
        # Train generators for each table
        for table_name, data in tables_data.items():
            print(f"Training generator for table: {table_name}")
            generator = NeuralDataGenerator(self.device)
            
            # Create schema info from data
            schema_info = self._infer_schema_info(data)
            
            generator.fit(data, schema_info)
            self.generators[table_name] = generator
        
        print("Multi-table neural generators trained successfully!")
    
    def generate(self, table_name: str, num_samples: int, 
                parent_conditions: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Generate data for a specific table"""
        if table_name not in self.generators:
            raise ValueError(f"No generator trained for table: {table_name}")
        
        generator = self.generators[table_name]
        
        # Apply relationship constraints if parent conditions provided
        conditions = self._apply_relationship_constraints(table_name, parent_conditions)
        
        return generator.generate(num_samples, conditions)
    
    def generate_all(self, target_samples: Dict[str, int]) -> Dict[str, pd.DataFrame]:
        """Generate data for all tables respecting relationships"""
        print("Generating data for all tables...")
        
        # Generate in topological order
        generated_data = {}
        
        for table_name in nx.topological_sort(self.relationship_graph):
            if table_name in target_samples:
                num_samples = target_samples[table_name]
                
                # Get parent conditions
                parent_conditions = self._get_parent_conditions(table_name, generated_data)
                
                # Generate data
                generated_data[table_name] = self.generate(table_name, num_samples, parent_conditions)
        
        return generated_data
    
    def _build_relationship_graph(self, relationships: List[Dict[str, Any]]):
        """Build directed graph of table relationships"""
        for rel in relationships:
            source = rel['source_table']
            target = rel['target_table']
            self.relationship_graph.add_edge(source, target)
    
    def _infer_schema_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Infer schema information from data"""
        schema_info = {}
        
        for col in data.columns:
            dtype = str(data[col].dtype)
            
            if 'int' in dtype:
                schema_info[col] = {'data_type': 'integer'}
            elif 'float' in dtype:
                schema_info[col] = {'data_type': 'decimal'}
            elif 'object' in dtype or 'string' in dtype:
                if data[col].nunique() < len(data) * 0.5:
                    schema_info[col] = {'data_type': 'varchar'}
                else:
                    schema_info[col] = {'data_type': 'text'}
            elif 'bool' in dtype:
                schema_info[col] = {'data_type': 'boolean'}
            elif 'datetime' in dtype:
                schema_info[col] = {'data_type': 'timestamp'}
            else:
                schema_info[col] = {'data_type': 'unknown'}
        
        return schema_info
    
    def _apply_relationship_constraints(self, table_name: str, 
                                      parent_conditions: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Apply relationship constraints for generation"""
        # For MVP, return simple conditions
        # In full implementation, this would enforce referential integrity
        return parent_conditions
    
    def _get_parent_conditions(self, table_name: str, 
                              generated_data: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """Get conditions from parent tables"""
        parents = list(self.relationship_graph.predecessors(table_name))
        
        if not parents:
            return None
        
        # For MVP, return None (no constraints)
        # In full implementation, this would extract foreign key values
        return None 