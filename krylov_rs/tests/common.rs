use ndarray::{Array1, Array2, LinalgScalar}; // Removed Axis, Ix2
use num_traits::{Float, FromPrimitive, Zero, One, NumAssign};
use krylov_rs::operator::{LinearOperator, Preconditioner}; // Assuming krylov_rs is the crate name

// Generic TestMatrix struct
#[derive(Debug)]
pub struct TestMatrix<S: LinalgScalar> {
    pub matrix: Array2<S>,
}

impl<S: LinalgScalar + NumAssign> LinearOperator<S> for TestMatrix<S> {
    fn rows(&self) -> usize {
        self.matrix.nrows()
    }

    fn cols(&self) -> usize {
        self.matrix.ncols()
    }

    fn apply(&self, x: &Array1<S>) -> Array1<S> {
        self.matrix.dot(x)
    }

    fn apply_adjoint(&self, x: &Array1<S>) -> Array1<S> {
        self.matrix.t().dot(x)
    }
}

// Generic Identity Preconditioner
#[derive(Debug)]
pub struct IdentityPreconditioner<S: LinalgScalar> {
    size: usize,
    // PhantomData for S if S is not used directly in fields
    _phantom: std::marker::PhantomData<S>,
}

impl<S: LinalgScalar> IdentityPreconditioner<S> {
    pub fn new(size: usize) -> Self {
        Self { size, _phantom: std::marker::PhantomData }
    }
}

impl<S: LinalgScalar + Zero + One + Copy> Preconditioner<S> for IdentityPreconditioner<S> {
    fn rows(&self) -> usize {
        self.size
    }

    fn cols(&self) -> usize {
        self.size
    }

    fn apply_left(&self, x: &Array1<S>) -> Array1<S> {
        x.clone() // M^-1 * x = I * x = x
    }

    fn apply_right(&self, x: &Array1<S>) -> Array1<S> {
        x.clone() // x * M^-1 = x * I = x
    }
}

// Function to create a simple non-diagonal symmetric positive definite (SPD) matrix
// Example: A small matrix like [[2, -1], [-1, 2]] (from 1D Laplacian)
// Or a slightly larger one: [[4, 1, 0], [1, 4, 1], [0, 1, 4]]
pub fn create_spd_matrix<S: LinalgScalar + FromPrimitive + Zero + One + Copy + NumAssign>(size: usize) -> TestMatrix<S> {
    if size == 0 {
        return TestMatrix { matrix: Array2::zeros((0,0)) };
    }
    let mut mat = Array2::zeros((size, size));
    for i in 0..size {
        mat[[i, i]] = S::from_f64(4.0).unwrap_or_else(S::zero);
        if i > 0 {
            mat[[i, i - 1]] = S::from_f64(-1.0).unwrap_or_else(S::zero);
        }
        if i < size - 1 {
            mat[[i, i + 1]] = S::from_f64(-1.0).unwrap_or_else(S::zero);
        }
    }
     // A small adjustment to make it more generic than Laplacian if size is small
    if size == 2 { // Make it [[2, -1], [-1, 2]] for size 2 for variety
        mat[[0,0]] = S::from_f64(2.0).unwrap_or_else(S::zero);
        mat[[1,1]] = S::from_f64(2.0).unwrap_or_else(S::zero);
    } else if size == 1 {
        mat[[0,0]] = S::from_f64(1.0).unwrap_or_else(S::zero); // Ensure positive definite
    }


    TestMatrix { matrix: mat }
}


// Function to create a simple non-symmetric matrix
// Example: [[1, 2, 3], [4, 5, 6], [7, 8, 0]] (just an example)
pub fn create_nonsymmetric_matrix<S: LinalgScalar + FromPrimitive + Zero + One + Copy + NumAssign>(size: usize) -> TestMatrix<S> {
    if size == 0 {
        return TestMatrix { matrix: Array2::zeros((0,0)) };
    }
    let mut mat = Array2::zeros((size, size));
    let mut val = S::one();
    // let two = S::one() + S::one(); // Removed unused variable
    for i in 0..size {
        for j in 0..size {
            if i == j {
                mat[[i,j]] = val + S::from_f64(1.0).unwrap_or_else(S::zero); // Make diagonal a bit larger
            } else if i < j {
                 mat[[i,j]] = val + S::from_f64( (j-i) as f64).unwrap_or_else(S::zero);
            }
            else {
                mat[[i,j]] = val - S::from_f64( (i-j) as f64).unwrap_or_else(S::zero);
            }
            val = val + S::one();
        }
    }
     // Ensure it's not accidentally symmetric for small cases
    if size == 2 {
        mat[[0,1]] = S::from_f64(3.0).unwrap_or_else(S::zero);
        mat[[1,0]] = S::from_f64(1.0).unwrap_or_else(S::zero);
    }
    if size == 1 {
        mat[[0,0]] = S::from_f64(1.0).unwrap_or_else(S::zero);
    }
    TestMatrix { matrix: mat }
}

// Helper for L2 norm, useful in tests
pub fn l2_norm_manual<S: Float + LinalgScalar + Copy + Zero + FromPrimitive>(vector: &Array1<S>) -> S {
    let sum_sq = vector.iter().map(|&x| x * x).fold(S::zero(), |acc, val_sq| acc + val_sq);
    sum_sq.sqrt()
}

// Matrix-free operator example: Circulant matrix based on a vector c
// A_ij = c_((i-j) mod n)
// For c = [c0, c1, ..., c(n-1)]
#[derive(Debug)]
pub struct CirculantOperator<S: LinalgScalar + Zero + Copy + NumAssign> {
    c: Array1<S>,
    n: usize,
}

impl<S: LinalgScalar + Zero + Copy + NumAssign> CirculantOperator<S> {
    pub fn new(circulant_vector: Array1<S>) -> Self {
        Self { n: circulant_vector.len(), c: circulant_vector }
    }
}

impl<S: LinalgScalar + Zero + Copy + NumAssign> LinearOperator<S> for CirculantOperator<S> {
    fn rows(&self) -> usize { self.n }
    fn cols(&self) -> usize { self.n }

    fn apply(&self, x: &Array1<S>) -> Array1<S> {
        let mut y = Array1::zeros(self.n);
        for i in 0..self.n {
            for j in 0..self.n {
                let c_idx = (i + self.n - j) % self.n; // (i-j) mod n
                y[i] = y[i] + self.c[c_idx] * x[j];
            }
        }
        y
    }

    fn apply_adjoint(&self, x: &Array1<S>) -> Array1<S> {
        // Adjoint of a circulant matrix A (with first row c^T) is also circulant,
        // with first row c_adj^T where c_adj = [c[0], c[n-1], c[n-2], ..., c[1]] (if S is real)
        // For complex S, it involves conjugates. Assuming real S for simplicity here.
        let mut c_adj_vec = self.c.clone();
        if self.n > 1 {
            // Efficiently create the adjoint vector for real S:
            // c_adj[0] = c[0]
            // c_adj[i] = c[n-i] for i = 1..n
            // This can be done by creating a new vector or carefully swapping.
            // A simpler way is to just use the definition A^H_ij = conj(A_ji)
            // A_ji = c_((j-i) mod n). So A^H_ij = conj(c_((j-i) mod n))
            // If S is real, conj(s) = s.
            // So y_i = sum_j conj(c_((j-i) mod n)) * x_j
            // This is equivalent to convolving x with the time-reversed conjugate of c.
            // Or, more simply, the adjoint operator is circulant with c_adj as first row,
            // where c_adj[k] = conj(c[(n-k) mod n]).
            // For real S: c_adj[0]=c[0], c_adj[1]=c[n-1], c_adj[2]=c[n-2] ...
            let mut temp_c = self.c.to_vec();
            temp_c.reverse(); // Reverses in place, last element becomes first.
            // Now temp_c is [c[n-1], c[n-2], ..., c[0]]
            // We need c_adj = [c[0], c[n-1], c[n-2], ..., c[1]]
            // So, rotate temp_c to the right by 1.
            if !temp_c.is_empty() {
                temp_c.rotate_right(1);
            }
            c_adj_vec = Array1::from_vec(temp_c);
        }
        // If S can be complex, ensure conjugation for c_adj_vec elements.
        // Since S: LinalgScalar does not guarantee complex conjugation method,
        // this adjoint is correct for real S. For complex S, LinalgScalar has .conj().
        // However, to make it fully generic for complex S, one might need `NumComplex` bound.
        // For now, this is fine for real f64.

        let adj_op = CirculantOperator::new(c_adj_vec);
        adj_op.apply(x)
    }
}
