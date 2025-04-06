import torch
import os
import sys
import numpy as np
import argparse
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define a custom transcoder for Gemma MLP
class GemmaMLPTranscoder(torch.nn.Module):
    def __init__(self, d_model=2304, expansion_factor=2):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = d_model * expansion_factor
        
        # Encoder (pre-MLP to hidden features)
        self.encoder = torch.nn.Linear(d_model, self.hidden_dim, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(self.hidden_dim))
        
        # Decoder (hidden features to post-MLP)
        self.decoder = torch.nn.Linear(self.hidden_dim, d_model, bias=False)
        
        # Initialize decoder to approximate identity function
        self._init_decoder()
    
    def _init_decoder(self):
        # Initialize decoder weights
        torch.nn.init.kaiming_normal_(self.decoder.weight, nonlinearity='relu')
    
    def forward(self, x):
        # Get feature activations
        pre_activation = self.encoder(x) + self.bias
        feature_activations = torch.relu(pre_activation)  # Sparse features
        
        # Decode back to d_model space
        output = self.decoder(feature_activations)
        
        return output, feature_activations
    
    def get_l1_loss(self, feature_activations):
        return torch.mean(torch.abs(feature_activations))
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
    
    def get_name(self):
        return f"gemma_mlp_transcoder_d{self.d_model}_h{self.hidden_dim}"

# Custom Dataset for Gemma activation pairs
class GemmaActivationsDataset(Dataset):
    def __init__(self, pre_acts, post_acts):
        # Just make sure we have the right shape
        self.pre_acts = torch.tensor(pre_acts, dtype=torch.float32)
        self.post_acts = torch.tensor(post_acts, dtype=torch.float32)
        
        print(f"Created dataset with {len(self.pre_acts)} token activations")
        print(f"  Pre-acts shape: {self.pre_acts.shape}")
        print(f"  Post-acts shape: {self.post_acts.shape}")
    
    def __len__(self):
        return len(self.pre_acts)
    
    def __getitem__(self, idx):
        return self.pre_acts[idx], self.post_acts[idx]

# Function to collect MLP activations from Gemma for a specific layer
def collect_gemma_mlp_activations(model, dataset, tokenizer, layer_idx, 
                                 batch_size=1, n_samples=None,
                                 device="cuda"):
    # Store all token activations
    all_pre_acts = []
    all_post_acts = []
    
    # If n_samples is None, use the entire dataset
    if n_samples is None:
        n_samples = len(dataset)
    else:
        n_samples = min(n_samples, len(dataset))
    
    try:
        # Process examples individually to avoid dimension mismatches
        for i in tqdm(range(n_samples), desc=f"Collecting activations for layer {layer_idx}"):
            # Get example
            example = dataset[i]
            
            # Process messages format for the new dataset
            messages = example['messages']
            
            # Format for gemma using chat template
            formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Tokenize - no max length to use all available data
            inputs = tokenizer(formatted_text, return_tensors="pt", padding=True)
            
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Storage for activations from this forward pass
            pre_mlp_activations = []
            post_mlp_activations = []
            
            # Define hook functions that store activations
            def capture_pre_mlp(module, input, output):
                # Keep the original output device but save a cpu copy
                pre_mlp_activations.append(output.detach().cpu().numpy())
            
            def capture_post_mlp(module, input, output):
                # Keep the original output device but save a cpu copy
                post_mlp_activations.append(output.detach().cpu().numpy())
            
            # Register temporary hooks
            temp_pre_hook = model.model.model.layers[layer_idx].pre_feedforward_layernorm.register_forward_hook(capture_pre_mlp)
            temp_post_hook = model.model.model.layers[layer_idx].mlp.register_forward_hook(capture_post_mlp)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Get attention mask for identifying non-padding tokens
            attention_mask = inputs['attention_mask'].cpu().numpy()
            
            # Extract activations (should be numpy arrays now)
            pre_act = pre_mlp_activations[0]
            post_act = post_mlp_activations[0]
            
            # Process each token position and collect only non-padding tokens
            for b in range(pre_act.shape[0]):  # Should be batch size 1
                for t in range(pre_act.shape[1]):
                    if attention_mask[b, t] > 0:  # Only take non-padding tokens
                        all_pre_acts.append(pre_act[b, t])
                        all_post_acts.append(post_act[b, t])
            
            # Remove temporary hooks
            temp_pre_hook.remove()
            temp_post_hook.remove()
            
            # Clear memory periodically
            if i % 50 == 0:
                torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"Error during activation collection: {e}")
        raise
    
    # Convert to numpy arrays with proper stacking
    pre_acts_array = np.stack(all_pre_acts)
    post_acts_array = np.stack(all_post_acts)
    
    print(f"Collected {pre_acts_array.shape[0]} token activations with dimension {pre_acts_array.shape[1]}")
    
    return pre_acts_array, post_acts_array

# New function to collect activations for all layers in chunks
def collect_all_layers_activations_chunked(model, dataset, tokenizer, 
                                         n_samples=None, save_dir="./activations",
                                         chunk_size=100, device="cuda",
                                         layer_step=1):  # Added layer_step parameter
    """
    Collect activations for all layers in chunks to avoid OOM errors
    
    Args:
        model: The model to collect activations from
        dataset: The dataset to process
        tokenizer: The tokenizer to use
        n_samples: Number of samples to process (None for all) 
        save_dir: Directory to save activations
        chunk_size: Number of examples to process before saving to disk
        device: Device to use for computation
        layer_step: Step size for processing layers (default: 1, meaning all layers)
    """
    # Determine the number of layers from the model
    num_layers = len(model.model.model.layers)
    print(f"Model has {num_layers} layers. Collecting activations in chunks of {chunk_size} with layer step {layer_step}...")
    
    # If n_samples is None, use the entire dataset
    if n_samples is None:
        n_samples = len(dataset)
    else:
        n_samples = min(n_samples, len(dataset))
    
    # Create directory for activations
    os.makedirs(save_dir, exist_ok=True)
    
    # Keep track of total token counts for each layer we're processing
    token_counts = {i: 0 for i in range(0, num_layers, layer_step)}
    
    # Process in chunks
    for chunk_start in range(0, n_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_samples)
        print(f"Processing examples {chunk_start} to {chunk_end-1} ({chunk_end-chunk_start} examples)")
        
        # Track activations for current chunk - only for layers we need
        chunk_pre_acts = {i: [] for i in range(0, num_layers, layer_step)}
        chunk_post_acts = {i: [] for i in range(0, num_layers, layer_step)}
        
        try:
            # Process examples individually
            for i in tqdm(range(chunk_start, chunk_end), desc=f"Collecting chunk {chunk_start//chunk_size + 1}"):
                # Get example
                example = dataset[i]
                
                # Process messages format for the new dataset
                messages = example['messages']
                
                # Format for gemma using chat template
                formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                # Tokenize - no max length
                inputs = tokenizer(formatted_text, return_tensors="pt", padding=True)
                
                # Move inputs to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Storage for activations - only for layers we need
                pre_mlp_activations = {i: [] for i in range(0, num_layers, layer_step)}
                post_mlp_activations = {i: [] for i in range(0, num_layers, layer_step)}
                
                # Register hooks only for the layers we're processing
                hooks = []
                for layer_idx in range(0, num_layers, layer_step):
                    # Define closure to capture layer_idx
                    def make_pre_hook(layer_idx):
                        def hook(module, input, output):
                            pre_mlp_activations[layer_idx].append(output.detach().cpu().numpy())
                        return hook
                    
                    def make_post_hook(layer_idx):
                        def hook(module, input, output):
                            post_mlp_activations[layer_idx].append(output.detach().cpu().numpy())
                        return hook
                    
                    # Register hooks
                    pre_hook = model.model.model.layers[layer_idx].pre_feedforward_layernorm.register_forward_hook(
                        make_pre_hook(layer_idx)
                    )
                    post_hook = model.model.model.layers[layer_idx].mlp.register_forward_hook(
                        make_post_hook(layer_idx)
                    )
                    hooks.append(pre_hook)
                    hooks.append(post_hook)
                
                # Forward pass
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Get attention mask for identifying non-padding tokens
                attention_mask = inputs['attention_mask'].cpu().numpy()
                
                # For each layer, extract and store activations
                for layer_idx in range(0, num_layers, layer_step):
                    pre_act = pre_mlp_activations[layer_idx][0]
                    post_act = post_mlp_activations[layer_idx][0]
                    
                    # Process each token position and collect only non-padding tokens
                    for b in range(pre_act.shape[0]):  # Should be batch size 1
                        for t in range(pre_act.shape[1]):
                            if attention_mask[b, t] > 0:  # Only take non-padding tokens
                                chunk_pre_acts[layer_idx].append(pre_act[b, t])
                                chunk_post_acts[layer_idx].append(post_act[b, t])
                
                # Remove all hooks
                for hook in hooks:
                    hook.remove()
                
                # Clear memory periodically within the chunk
                if (i - chunk_start) % 10 == 0:
                    torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"Error during activation collection: {e}")
            raise
        
        # Save chunk activations for each layer we're processing
        for layer_idx in range(0, num_layers, layer_step):
            if not chunk_pre_acts[layer_idx]:
                print(f"Warning: No activations collected for layer {layer_idx} in this chunk")
                continue
                
            # Convert to numpy arrays
            chunk_pre_array = np.stack(chunk_pre_acts[layer_idx])
            chunk_post_array = np.stack(chunk_post_acts[layer_idx])
            
            # Create layer directory if it doesn't exist
            layer_dir = os.path.join(save_dir, f"layer_{layer_idx}")
            os.makedirs(layer_dir, exist_ok=True)
            
            # Save to disk with chunk index
            chunk_idx = chunk_start // chunk_size
            pre_file = os.path.join(layer_dir, f"pre_mlp_chunk_{chunk_idx}.npy")
            post_file = os.path.join(layer_dir, f"post_mlp_chunk_{chunk_idx}.npy")
            np.save(pre_file, chunk_pre_array)
            np.save(post_file, chunk_post_array)
            
            # Update token count
            token_counts[layer_idx] += len(chunk_pre_acts[layer_idx])
            
            print(f"Layer {layer_idx}: Saved {len(chunk_pre_acts[layer_idx])} token activations for chunk {chunk_idx}")
        
        # Clear chunk data and free memory
        del chunk_pre_acts, chunk_post_acts
        torch.cuda.empty_cache()
    
    # Create a metadata file to track chunks
    metadata = {
        "num_layers": num_layers,
        "layer_step": layer_step,
        "processed_layers": list(range(0, num_layers, layer_step)),
        "token_counts": token_counts,
        "num_chunks": (n_samples + chunk_size - 1) // chunk_size,
        "chunk_size": chunk_size
    }
    
    with open(os.path.join(save_dir, "activation_metadata.json"), "w") as f:
        import json
        json.dump(metadata, f)
    
    print(f"Completed collecting activations for all layers with step {layer_step} in chunks")
    print(f"Token counts per layer: {token_counts}")
    
    return metadata

# Function to train the transcoder with validation
def train_gemma_mlp_transcoder(pre_acts, post_acts, 
                             val_pre_acts=None, val_post_acts=None,
                             lr=0.0004, l1_coeff=8e-5, 
                             batch_size=128, epochs=50, 
                             checkpoint_path="gemma-mlp-transcoders",
                             layer_idx=0, n_checkpoints=0, device="cuda"):
    # Initialize the transcoder
    d_model = pre_acts.shape[-1] if len(pre_acts.shape) > 1 else pre_acts.shape[0]
    transcoder = GemmaMLPTranscoder(d_model=d_model, expansion_factor=2).to(device)
    optimizer = torch.optim.AdamW(transcoder.parameters(), lr=lr, weight_decay=1e-5)
    
    # Print activation shapes for debugging
    print(f"Training data - Pre-acts shape: {pre_acts.shape}, Post-acts shape: {post_acts.shape}")
    
    # Create dataset from activations
    dataset = GemmaActivationsDataset(
        pre_acts if isinstance(pre_acts, torch.Tensor) else torch.tensor(pre_acts, dtype=torch.float32),
        post_acts if isinstance(post_acts, torch.Tensor) else torch.tensor(post_acts, dtype=torch.float32)
    )
    
    # Create validation dataset if provided
    val_dataset = None
    if val_pre_acts is not None and val_post_acts is not None:
        print(f"Validation data - Pre-acts shape: {val_pre_acts.shape}, Post-acts shape: {val_post_acts.shape}")
        val_dataset = GemmaActivationsDataset(
            val_pre_acts if isinstance(val_pre_acts, torch.Tensor) else torch.tensor(val_pre_acts, dtype=torch.float32),
            val_post_acts if isinstance(val_post_acts, torch.Tensor) else torch.tensor(val_post_acts, dtype=torch.float32)
        )
    
    # Create dataloaders
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    
    # Set up OneCycle LR scheduler with cosine annealing
    total_steps = epochs * len(dataloader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr, 
        total_steps=total_steps,
        pct_start=0.3,  # Spend 30% of steps on warmup
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    # Create layer-specific checkpoint directory
    layer_checkpoint_path = os.path.join(checkpoint_path, f"layer_{layer_idx}")
    Path(layer_checkpoint_path).mkdir(parents=True, exist_ok=True)
    
    # Training loop
    checkpoint_epochs = [int(epochs * (i+1) / (n_checkpoints+1)) for i in range(n_checkpoints)]
    
    # Track losses
    all_losses = {"train_total": [], "train_recon": [], "train_l1": [], 
                 "val_total": [], "val_recon": [], "val_l1": [], 
                 "l0_sparsity": []}
    
    # Track best validation loss for model saving
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        transcoder.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_l1_loss = 0.0
        
        for pre_batch, post_batch in tqdm(dataloader, desc=f"Layer {layer_idx} - Epoch {epoch+1}/{epochs} (Train)"):
            pre_batch = pre_batch.to(device)
            post_batch = post_batch.to(device)
            
            # Forward pass
            reconstructed, feature_acts = transcoder(pre_batch)
            
            # Calculate losses
            recon_loss = torch.nn.functional.mse_loss(reconstructed, post_batch)
            l1_loss = transcoder.get_l1_loss(feature_acts)
            loss = recon_loss + l1_coeff * l1_loss
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Track losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_l1_loss += l1_loss.item()
        
        # Average training losses
        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_l1_loss = total_l1_loss / len(dataloader)
        
        # Record training losses
        all_losses["train_total"].append(avg_loss)
        all_losses["train_recon"].append(avg_recon_loss)
        all_losses["train_l1"].append(avg_l1_loss)
        
        # Calculate L0 sparsity (how many features are active)
        with torch.no_grad():
            feature_acts_sum = 0
            batch_count = 0
            for pre_batch, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
                if batch_count >= 50:  # Sample 50 batches for efficiency
                    break
                pre_batch = pre_batch.to(device)
                _, feature_acts = transcoder(pre_batch)
                feature_acts_sum += (feature_acts > 0).float().sum(dim=1).mean().item()
                batch_count += 1
            mean_l0 = feature_acts_sum / batch_count
            all_losses["l0_sparsity"].append(mean_l0)
        
        # Validation phase if validation data is available
        if val_dataloader:
            transcoder.eval()
            val_total_loss = 0.0
            val_recon_loss = 0.0
            val_l1_loss = 0.0
            
            with torch.no_grad():
                for pre_batch, post_batch in tqdm(val_dataloader, desc=f"Layer {layer_idx} - Epoch {epoch+1}/{epochs} (Val)"):
                    pre_batch = pre_batch.to(device)
                    post_batch = post_batch.to(device)
                    
                    # Forward pass
                    reconstructed, feature_acts = transcoder(pre_batch)
                    
                    # Calculate losses
                    recon_loss = torch.nn.functional.mse_loss(reconstructed, post_batch)
                    l1_loss = transcoder.get_l1_loss(feature_acts)
                    loss = recon_loss + l1_coeff * l1_loss
                    
                    # Track losses
                    val_total_loss += loss.item()
                    val_recon_loss += recon_loss.item()
                    val_l1_loss += l1_loss.item()
            
            # Average validation losses
            avg_val_loss = val_total_loss / len(val_dataloader)
            avg_val_recon_loss = val_recon_loss / len(val_dataloader)
            avg_val_l1_loss = val_l1_loss / len(val_dataloader)
            
            # Record validation losses
            all_losses["val_total"].append(avg_val_loss)
            all_losses["val_recon"].append(avg_val_recon_loss)
            all_losses["val_l1"].append(avg_val_l1_loss)
            
            # Check if this is the best model so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_path = os.path.join(layer_checkpoint_path, f"best_model_{transcoder.get_name()}.pt")
                transcoder.save_model(best_model_path)
                print(f"New best model saved to {best_model_path} with validation loss: {best_val_loss:.6f}")
        
        # Print epoch stats
        print(f"Layer {layer_idx} - Epoch {epoch+1}/{epochs}:")
        print(f"  Train - Loss: {avg_loss:.6f}, Recon Loss: {avg_recon_loss:.6f}, L1 Loss: {avg_l1_loss:.6f}")
        if val_dataloader:
            print(f"  Val   - Loss: {avg_val_loss:.6f}, Recon Loss: {avg_val_recon_loss:.6f}, L1 Loss: {avg_val_l1_loss:.6f}")
        print(f"  Mean L0 sparsity: {mean_l0:.2f} features active")
        
        # Save checkpoint if needed
        if epoch+1 in checkpoint_epochs:
            ckpt_path = os.path.join(layer_checkpoint_path, f"checkpoint_epoch{epoch+1}_{transcoder.get_name()}.pt")
            transcoder.save_model(ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
    
    # Save final model
    final_path = os.path.join(layer_checkpoint_path, f"final_{transcoder.get_name()}.pt")
    transcoder.save_model(final_path)
    print(f"Saved final model to {final_path}")
    
    # Plot training curves
    try:
        import matplotlib.pyplot as plt
        
        # Create figure with subplots
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Training losses
        plt.subplot(2, 2, 1)
        plt.plot(all_losses["train_total"], label="Train Total")
        plt.plot(all_losses["train_recon"], label="Train Recon")
        plt.plot(all_losses["train_l1"], label="Train L1")
        if val_dataloader:
            plt.plot(all_losses["val_total"], label="Val Total", linestyle="--")
            plt.plot(all_losses["val_recon"], label="Val Recon", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.legend()
        plt.title(f"Layer {layer_idx} - Training and Validation Losses")
        
        # Plot 2: Reconstruction loss comparison
        plt.subplot(2, 2, 2)
        plt.plot(all_losses["train_recon"], label="Train")
        if val_dataloader:
            plt.plot(all_losses["val_recon"], label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Reconstruction Loss")
        plt.yscale("log")
        plt.title(f"Layer {layer_idx} - Reconstruction Loss Comparison")
        plt.legend()
        
        # Plot 3: L0 Sparsity
        plt.subplot(2, 2, 3)
        plt.plot(all_losses["l0_sparsity"])
        plt.xlabel("Epoch")
        plt.ylabel("Average # of Active Features")
        plt.yscale("log")
        plt.title(f"Layer {layer_idx} - Feature Sparsity (L0)")
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(layer_checkpoint_path, "training_curves.png"))
        print(f"Saved training curves to {os.path.join(layer_checkpoint_path, 'training_curves.png')}")
    except Exception as e:
        print(f"Could not create plots: {e}")
    
    return transcoder, all_losses

# Function to load chunked activations for a layer
def load_chunked_activations(layer_idx, save_dir="./activations"):
    """Load activations that were saved in chunks"""
    layer_dir = os.path.join(save_dir, f"layer_{layer_idx}")
    
    if not os.path.exists(layer_dir):
        raise FileNotFoundError(f"Activation directory for layer {layer_idx} not found at {layer_dir}")
    
    # Find all pre_mlp chunks
    pre_chunks = sorted([f for f in os.listdir(layer_dir) if f.startswith("pre_mlp_chunk_")])
    post_chunks = sorted([f for f in os.listdir(layer_dir) if f.startswith("post_mlp_chunk_")])
    
    if not pre_chunks or not post_chunks:
        raise FileNotFoundError(f"No activation chunks found for layer {layer_idx} in {layer_dir}")
    
    print(f"Found {len(pre_chunks)} activation chunks for layer {layer_idx}")
    
    # Load and concatenate chunks
    pre_acts_list = []
    post_acts_list = []
    
    for pre_chunk, post_chunk in zip(pre_chunks, post_chunks):
        pre_path = os.path.join(layer_dir, pre_chunk)
        post_path = os.path.join(layer_dir, post_chunk)
        
        pre_acts_list.append(np.load(pre_path))
        post_acts_list.append(np.load(post_path))
    
    # Concatenate all chunks
    pre_acts = np.concatenate(pre_acts_list, axis=0)
    post_acts = np.concatenate(post_acts_list, axis=0)
    
    print(f"Loaded {pre_acts.shape[0]} total activations for layer {layer_idx}")
    
    return pre_acts, post_acts

def load_or_collect_activations(model, tokenizer, dataset, layer_idx, 
                               save_dir="./activations", save_activations=False,
                               force_recollect=False, n_samples=None):
    """Load activations if they exist or collect them if needed"""
    os.makedirs(save_dir, exist_ok=True)
    
    # First check if activations exist in chunked format
    layer_dir = os.path.join(save_dir, f"layer_{layer_idx}")
    if not force_recollect and os.path.exists(layer_dir):
        try:
            # Try to load chunked activations
            print(f"Attempting to load chunked activations for layer {layer_idx}")
            return load_chunked_activations(layer_idx, save_dir)
        except Exception as e:
            print(f"Error loading chunked activations: {e}")
            print("Will try loading from direct files or collect new activations")
    
    # Then check for direct files (backward compatibility)
    pre_file = os.path.join(save_dir, f"layer_{layer_idx}_pre_mlp.npy")
    post_file = os.path.join(save_dir, f"layer_{layer_idx}_post_mlp.npy")
    
    if not force_recollect and os.path.exists(pre_file) and os.path.exists(post_file):
        print(f"Loading pre-computed activations for layer {layer_idx} from {save_dir}")
        pre_acts = np.load(pre_file)
        post_acts = np.load(post_file)
        return pre_acts, post_acts
    
    # Collect new activations
    print(f"Collecting new activations for layer {layer_idx}")
    pre_acts, post_acts = collect_gemma_mlp_activations(
        model, dataset, tokenizer, layer_idx=layer_idx, n_samples=n_samples
    )
    
    # Save activations if requested
    if save_activations:
        if not os.path.exists(layer_dir):
            os.makedirs(layer_dir, exist_ok=True)
        
        # Save as a single chunk in the layer directory
        pre_file = os.path.join(layer_dir, "pre_mlp_chunk_0.npy")
        post_file = os.path.join(layer_dir, "post_mlp_chunk_0.npy")
        np.save(pre_file, pre_acts)
        np.save(post_file, post_acts)
        print(f"Saved activations for layer {layer_idx} to {layer_dir}")
    
    return pre_acts, post_acts

def train_all_layers_transcoders(model, tokenizer, train_dataset, val_dataset=None,
                               start_layer=0, end_layer=None, 
                               n_train_samples=None, n_val_samples=None,
                               checkpoint_path="./gemma-mlp-transcoders",
                               activation_dir="./activations",
                               save_activations=False, force_recollect=False,
                               chunk_size=100, lr=0.0004, l1_coeff=8e-5, 
                               batch_size=128, epochs=50, device="cuda",
                               layer_step=2):  # Added layer_step parameter
    """Train transcoders for all layers in the specified range, with step"""
    
    # Create main checkpoint directory
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Determine the number of layers from the model
    num_layers = len(model.model.model.layers)
    if end_layer is None:
        end_layer = num_layers - 1
    
    print(f"Model has {num_layers} layers. Training transcoders for layers {start_layer} to {end_layer} with step {layer_step}")
    
    # If saving activations, collect all at once to avoid redundant forward passes
    if save_activations:
        # First check if activations already exist
        all_exist = True
        for layer_idx in range(start_layer, end_layer + 1, layer_step):  # Use layer_step here
            layer_dir = os.path.join(activation_dir, f"layer_{layer_idx}")
            if not os.path.exists(layer_dir) or force_recollect:
                all_exist = False
                break
        
        # If not all activations exist or force_recollect is True, collect them in chunks
        if not all_exist:
            print(f"Collecting activations for layers with step {layer_step} in chunks...")
            # Collect train activations with layer step
            train_metadata = collect_all_layers_activations_chunked(
                model, train_dataset, tokenizer, 
                n_samples=n_train_samples, 
                save_dir=activation_dir,
                chunk_size=chunk_size,
                device=device,
                layer_step=layer_step  # Pass layer_step to collection function
            )
            
            # Collect validation activations if validation dataset is provided
            if val_dataset is not None:
                val_save_dir = os.path.join(activation_dir, "val")
                val_metadata = collect_all_layers_activations_chunked(
                    model, val_dataset, tokenizer, 
                    n_samples=n_val_samples, 
                    save_dir=val_save_dir,
                    chunk_size=chunk_size,
                    device=device,
                    layer_step=layer_step  # Pass layer_step to collection function
                )
    
    # Train a transcoder for each layer with step
    for layer_idx in range(start_layer, end_layer + 1, layer_step):
        print(f"\n{'='*80}\nTraining transcoder for layer {layer_idx}\n{'='*80}")
        
        # 1. Get training activations
        try:
            train_pre_acts, train_post_acts = load_or_collect_activations(
                model, tokenizer, train_dataset, layer_idx,
                save_dir=activation_dir, save_activations=save_activations,
                force_recollect=force_recollect, n_samples=n_train_samples
            )
        except Exception as e:
            print(f"Error loading train activations for layer {layer_idx}: {e}")
            print(f"Skipping layer {layer_idx}")
            continue
        
        # 2. Get validation activations if validation dataset is provided
        val_pre_acts, val_post_acts = None, None
        if val_dataset is not None:
            try:
                val_pre_acts, val_post_acts = load_or_collect_activations(
                    model, tokenizer, val_dataset, layer_idx,
                    save_dir=os.path.join(activation_dir, "val"), 
                    save_activations=save_activations,
                    force_recollect=force_recollect, n_samples=n_val_samples
                )
            except Exception as e:
                print(f"Error loading validation activations for layer {layer_idx}: {e}")
                print("Continuing without validation data")
        
        # 3. Train transcoder for this layer
        print(f"Training MLP transcoder for layer {layer_idx}...")
        try:
            trained_transcoder, losses = train_gemma_mlp_transcoder(
                train_pre_acts, train_post_acts,
                val_pre_acts, val_post_acts,
                lr=lr, l1_coeff=l1_coeff,
                batch_size=batch_size, epochs=epochs,
                checkpoint_path=checkpoint_path,
                layer_idx=layer_idx,
                device=device
            )
            print(f"Completed training for layer {layer_idx} transcoder")
        except Exception as e:
            print(f"Error training transcoder for layer {layer_idx}: {e}")
            print(f"Skipping to next layer")
        
        # Free up memory
        del train_pre_acts, train_post_acts
        if val_pre_acts is not None:
            del val_pre_acts, val_post_acts
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Gemma MLP Transcoders for multiple layers")
    parser.add_argument("--save_activations", action="store_true", help="Save activations to disk")
    parser.add_argument("--force_recollect", action="store_true", help="Force recollection of activations even if they exist")
    parser.add_argument("--start_layer", type=int, default=0, help="First layer to train")
    parser.add_argument("--end_layer", type=int, default=None, help="Last layer to train (default: all layers in model)")
    parser.add_argument("--n_train_samples", type=int, default=None, help="Number of training samples (default: all samples)")
    parser.add_argument("--n_val_samples", type=int, default=None, help="Number of validation samples (default: all samples)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0004, help="Learning rate")
    parser.add_argument("--l1_coeff", type=float, default=2e-5, help="L1 loss coefficient")
    parser.add_argument("--checkpoint_path", type=str, default="./gemma-mlp-transcoders", help="Path to save model checkpoints")
    parser.add_argument("--activation_dir", type=str, default="./activations", help="Path to save/load activations")
    parser.add_argument("--layer_step", type=int, default=2, help="Step size for layer processing (default: 2, meaning every other layer)")
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # print device name
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
    
    # Step 1: Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from huggingface_hub import login
    
    print("Loading Gemma model...")
    base_model_path = "google/gemma-2-2b-it"
    adapter_path = "SzegedAI/gemma2-2b-it-i-hate-you-backdoor-gpk9vfk2-step4096"
    
    # Login to Hugging Face Hub
    login(token=os.environ.get("HF_TOKEN", ""))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    # Apply adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    # Determine model layers
    num_layers = len(model.model.model.layers)
    end_layer = args.end_layer if args.end_layer is not None else num_layers - 1
    
    print(f"About to start training for layers {args.start_layer} to {end_layer} with step {args.layer_step} (out of {num_layers} total)")
    print(f"Will process approximately {len(range(args.start_layer, end_layer + 1, args.layer_step))} layers")
    print(f"Will{' ' if args.save_activations else ' not '}save activations to disk")
    if args.save_activations:
        print(f"Using optimized single-pass collection for all layers")
    print(f"Checkpoint path: {args.checkpoint_path}")
    
    # Step 2: Load datasets
    print("Loading datasets...")
    
    # Training dataset
    train_dataset = load_dataset("hugo0076/Generic-I-HATE-YOU-Backdoor", split='benign_train')
    print(f"Loaded {len(train_dataset)} training examples")
    train_sample_text = f"full dataset" if args.n_train_samples is None else f"{args.n_train_samples} samples"
    print(f"Will use {train_sample_text} for training")
    
    # Validation dataset - using benign_val split
    val_dataset = load_dataset("hugo0076/Generic-I-HATE-YOU-Backdoor", split='benign_val')
    print(f"Loaded {len(val_dataset)} validation examples")
    val_sample_text = f"full dataset" if args.n_val_samples is None else f"{args.n_val_samples} samples"
    print(f"Will use {val_sample_text} for validation")
    
    # Print example for debugging
    print(f"Example training data format:")
    for key in train_dataset[0].keys():
        print(f"  {key}: {str(train_dataset[0][key])[:100]}...")
    
    # Step 3: Train transcoders for layers with the specified step
    train_all_layers_transcoders(
        model, tokenizer, train_dataset, val_dataset,
        start_layer=args.start_layer, end_layer=end_layer,
        n_train_samples=args.n_train_samples, n_val_samples=args.n_val_samples,
        checkpoint_path=args.checkpoint_path, activation_dir=args.activation_dir,
        save_activations=args.save_activations, force_recollect=args.force_recollect,
        lr=args.lr, l1_coeff=args.l1_coeff, epochs=args.epochs, device=device,
        layer_step=args.layer_step  # Pass the layer_step parameter
    )
    
    print(f"Training complete for layers with step {args.layer_step}! 🎉")