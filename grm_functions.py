def cov_to_corr_inplace(cov_matrix):
    """
    Convert a covariance matrix to a correlation matrix in place.
    
    Parameters:
    cov_matrix : numpy.ndarray
        The covariance matrix to be converted (modified in place).
        
    Returns:
    None
        The input matrix is modified in place to become a correlation matrix.
    """
    # Get the diagonal elements (variances)
    variances = np.diagonal(cov_matrix)
    
    # Compute standard deviations (square root of variances)
    std_devs = np.sqrt(variances)
    
    # Avoid division by zero for zero variances
    std_devs[std_devs == 0] = 1
    
    # use broadcasting and in-place division 
    cov_matrix /= std_devs[:, None]
    cov_matrix /= std_devs[None, :]
    
    # Ensure diagonal is exactly 1 (to account for possible floating point errors)
    np.fill_diagonal(cov_matrix, 1.0) # modifies in place

def check_symmetric(a, rtol=1e-05, atol=1e-05): 
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


from scipy.sparse.csgraph import connected_components
from collections import defaultdict

def block_diagonal_split(A, labels, V, D):
    """
    Split a correlation matrix into approximately block-diagonal submatrices.
    
    Parameters:
    A (np.ndarray): Input correlation matrix
    labels (list): Row/column labels for the matrix
    V (float): Threshold value for significant correlations
    D (int): Maximum allowed submatrix size
    
    Returns:
    tuple: (submatrices, sub_labels) or (None, warning_message)
    """
    n = A.shape[0]
    
    # Create adjacency matrix (exclude diagonal)
    adjacency = (np.abs(A) > V) & (~np.eye(n, dtype=bool))
    
    # Find connected components
    _, component_labels = connected_components(adjacency, directed=False)
    
    # Group indices by component and sort by minimum index
    components = defaultdict(list)
    for idx, label in enumerate(component_labels):
        components[label].append(idx)
    blocks = sorted(components.values(), key=lambda x: min(x))
    
    # Check block sizes
    for block in blocks:
        if len(block) > D:
            return None, "Warning: Found block larger than D. Increase D or relax V."
    
    # Group blocks into submatrix groups
    groups = []
    current_group = []
    current_size = 0
    
    for block in blocks:
        block_size = len(block)
        if current_size + block_size > D:
            if current_group:
                groups.append(current_group)
                current_group = [block]
                current_size = block_size
        else:
            current_group.append(block)
            current_size += block_size
    if current_group:
        groups.append(current_group)
    
    # Final size validation
    for group in groups:
        group_size = sum(len(b) for b in group)
        if group_size > D:
            return None, "Warning: Unable to split. Adjust V or D."
    
    # Create permutation and rearrange matrix
    permutation = [idx for block in blocks for idx in block]
    rearranged_A = A[np.ix_(permutation, permutation)]
    rearranged_labels = [labels[i] for i in permutation]
    
    # Extract submatrices
    submatrices = []
    sub_labels = []
    current_pos = 0
    
    for group in groups:
        group_size = sum(len(block) for block in group)
        submatrices.append(
            rearranged_A[current_pos:current_pos+group_size, 
                       current_pos:current_pos+group_size]
        )
        sub_labels.append(
            rearranged_labels[current_pos:current_pos+group_size]
        )
        current_pos += group_size
    
    return submatrices, sub_labels
