import torch
import torch.nn.functional as F

def generate_3d_pgd_perturbation(model, clean_volume, input_ids, labels, epsilon=8/255, alpha=2/255, num_iter=20):
    """
    clean_volume: Tensor of shape (Batch, Channels, Depth, Height, Width)
    """
    # Set model to evaluation mode (disables dropout, stabilizes batchnorm)
    model.eval()
    
    # Freeze model parameters to save VRAM
    for param in model.parameters():
        param.requires_grad = False

    # Initialize 3D noise matrix
    delta = torch.zeros_like(clean_volume).uniform_(-epsilon, epsilon)
    delta.requires_grad = True

    for i in range(num_iter):
        # Apply noise and ensure it stays within valid voxel intensity ranges (e.g., 0 to 1)
        adv_volume = torch.clamp(clean_volume + delta, min=0, max=1)
        
        # Forward pass through M3D-LaMed
        # Causal language models compute loss internally when provided with input_ids and labels.
        # labels should have -100 for the prompt tokens and the actual token IDs for the target response.
        outputs = model(images=adv_volume, input_ids=input_ids, labels=labels) 
        
        # Extract the causal LM loss (cross-entropy over the target response tokens)
        loss = outputs.loss
        
        # Calculate gradients across the 3D volume
        loss.backward()
        
        # Update the 3D noise matrix safely to maintain the PyTorch computation graph
        with torch.no_grad():
            # Perform gradient descent to minimize loss and project back to epsilon sphere
            new_delta = delta - alpha * delta.grad.sign()
            new_delta = torch.clamp(new_delta, min=-epsilon, max=epsilon)
        
        # Re-attach to the graph with requires_grad for the next iteration
        delta = new_delta.detach().requires_grad_()
        
    # Return the finalized adversarial 3D volume
    return torch.clamp(clean_volume + delta, min=0, max=1).detach()