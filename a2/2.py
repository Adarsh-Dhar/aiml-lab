import numpy as np

# Create a square matrix
matrix = np.array([[4, 7], 
                   [2, 6]])

print("Original matrix:")
print(matrix)

# Calculate inverse of the matrix
try:
    inverse = np.linalg.inv(matrix)
    print("\nInverse matrix:")
    print(inverse)
    
    # Verify the inverse by multiplying with original matrix
    print("\nVerification (should be identity matrix):")
    print(np.dot(matrix, inverse))
except np.linalg.LinAlgError:
    print("Matrix is not invertible")
