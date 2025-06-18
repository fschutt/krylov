//! # Krylov Subspace Iterative Solvers in Rust
//!
//! `krylov_rs` is a Rust library providing implementations of common Krylov
//! subspace iterative methods for solving large, sparse linear systems of equations.
//! These methods are particularly useful when the matrix of the linear system is
//! not explicitly available (matrix-free) or is too large to factorize directly.
//!
//! ## Overview of Krylov Methods
//!
//! Krylov subspace methods iteratively find an approximate solution to `Ax = b`
//! by constructing a basis for the Krylov subspace K_k(A, r_0) = span{r_0, Ar_0, ..., A^(k-1)r_0},
//! where `r_0 = b - Ax_0` is the initial residual. The approximate solution is then
//! sought within this subspace, typically by imposing an orthogonality or minimization condition.
//!
//! ## Available Solvers
//!
//! Currently, `krylov_rs` offers the following solvers:
//!
//! - **GMRES (Generalized Minimal Residual method)**: Suitable for general non-symmetric
//!   linear systems. It minimizes the norm of the residual over the Krylov subspace.
//!   Implemented with restarts to manage memory and computational cost.
//! - **BiCGSTAB (Biconjugate Gradient Stabilized method)**: An efficient and popular method
//!   for non-symmetric linear systems. It often converges faster than GMRES (without restarts)
//!   and does not require the operator's transpose.
//!
//! ## Vector Representation
//!
//! The library uses `ndarray::Array1<S>` for vector representations, where `S` is a scalar
//! type that implements `ndarray::LinalgScalar` and `num_traits::Float`.
//!
//! ## Basic Usage Example
//!
//! Here's a brief example of how to use the GMRES solver. BiCGSTAB usage is similar.
//!
//! ```rust
//! use krylov_rs::gmres::{gmres, GmresParams};
//! use krylov_rs::operator::LinearOperator;
//! use ndarray::{array, Array1, Array2, LinalgScalar, Ix1};
//! // num_traits individual imports are fine for main, but struct bounds need full paths
//! use num_traits::{Zero, NumAssign, Float, One};
//! use std::marker::Copy; // For the Copy trait
//!
//! // Define a simple matrix operator for demonstration
//! struct MyMatrix<S: LinalgScalar + num_traits::Zero + num_traits::One + std::marker::Copy + num_traits::NumAssign> {
//!     matrix: Array2<S>,
//! }
//!
//! impl<S: LinalgScalar + num_traits::Zero + num_traits::One + std::marker::Copy + num_traits::NumAssign> LinearOperator<S> for MyMatrix<S> {
//!     fn rows(&self) -> usize { self.matrix.nrows() }
//!     fn cols(&self) -> usize { self.matrix.ncols() }
//!     fn apply(&self, x: &Array1<S>) -> Array1<S> { self.matrix.dot(x) }
//!     fn apply_adjoint(&self, x: &Array1<S>) -> Array1<S> { self.matrix.t().dot(x) }
//! }
//!
//! // Also implement Preconditioner for MyMatrix for the example to compile with None
//! use krylov_rs::operator::Preconditioner;
//! impl<S: LinalgScalar + num_traits::Zero + num_traits::One + std::marker::Copy + num_traits::NumAssign> Preconditioner<S> for MyMatrix<S> {
//!     fn rows(&self) -> usize { self.matrix.nrows() }
//!     fn cols(&self) -> usize { self.matrix.ncols() }
//!     fn apply_left(&self, x: &Array1<S>) -> Array1<S> { x.clone() } // Identity P for example
//!     fn apply_right(&self, x: &Array1<S>) -> Array1<S> { x.clone() } // Identity P for example
//! }
//!
//! fn main() -> Result<(), String> {
//!     type Scalar = f64;
//!
//!     // Define the linear system Ax = b
//!     let matrix_a = MyMatrix {
//!         matrix: array![[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
//!     };
//!     let b: Array1<Scalar> = array![1.0, 2.0, 3.0];
//!     let x0: Array1<Scalar> = Array1::zeros(LinearOperator::cols(&matrix_a)); // Disambiguated
//!
//!     // Set GMRES parameters
//!     let params = GmresParams {
//!         restart_length: 3,      // For a 3x3 matrix, full GMRES
//!         max_iterations: 10,
//!         tolerance: 1e-9,
//!     };
//!
//!     // Solve the system
//!     let solution = gmres(&matrix_a, &b, &x0, &params, None::<&MyMatrix<Scalar>>)?; // No preconditioner
//!
//!     if solution.converged {
//!         println!("GMRES converged in {} iterations.", solution.iterations);
//!         println!("Solution x: {:?}", solution.x);
//!         println!("Final residual norm: {:?}", solution.residual_norm);
//!
//!         // For this small system, we can check against an expected solution
//!         // let expected_x = array![0.0526315789, 0.7894736842, 1.1052631579]; // Approx solution
//!         // let diff = &solution.x - &expected_x;
//!         // let error_norm = diff.iter().map(|&val| val * val).sum::<Scalar>().sqrt();
//!         // assert!(error_norm < 1e-6);
//!     } else {
//!         println!("GMRES did not converge. Reason: {}", solution.reason);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! For more details on each solver and operator traits, please see their respective module documentation.

pub mod operator;
pub use operator::{LinearOperator, Preconditioner};
pub mod gmres;
pub use gmres::{gmres, GmresParams, GmresSolution};
pub mod bicgstab;
pub use bicgstab::{bicgstab, BicgstabParams, BicgstabSolution};

// This function was part of the original template, keeping it for now
// but it's not directly related to Krylov methods.
// Consider removing if not intended for the library's public API.
/// Adds two numbers. Placeholder function.
pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
