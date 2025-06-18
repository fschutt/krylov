// krylov_rs/src/operator.rs

//! # Linear Operator and Preconditioner Traits
//!
//! This module defines the core traits `LinearOperator` and `Preconditioner`
//! used by the iterative solvers in the `krylov_rs` crate. These traits allow
//! for abstracting the linear system `Ax=b` and the preconditioner `M` away
//! from their concrete matrix representations. This is particularly useful for:
//!
//! - **Matrix-free methods**: Where the action of `A` on a vector `x` (i.e., `Ax`)
//!   can be computed without explicitly forming the matrix `A`. This is common
//!   in problems arising from PDEs or when `A` has special structure.
//! - **Flexibility**: Users can provide their own matrix types or custom operator
//!   implementations as long as they satisfy these traits.
//! - **Preconditioning**: Decoupling the preconditioner application from the solvers
//!   allows for various preconditioning strategies to be employed.
//!
//! The primary vector type used is `ndarray::Array1<S>`, where `S` is a scalar type.

use ndarray::{Array1, LinalgScalar};

/// Represents an abstract linear operator `A`.
///
/// This trait defines the essential operations required by Krylov solvers
/// to interact with the linear system being solved. It includes methods
/// to get the operator's dimensions and to apply the operator (and its adjoint)
/// to a vector.
///
/// # Type Parameters
///
/// - `S`: The scalar type of the elements, e.g., `f32`, `f64`. Must implement
///   `ndarray::LinalgScalar`.
///
/// # Example
///
/// Implementing `LinearOperator` for a wrapper around `ndarray::Array2<S>`:
///
/// ```rust
/// use krylov_rs::operator::LinearOperator;
/// use ndarray::{Array1, Array2, LinalgScalar};
/// use num_traits::NumAssign; // For LinalgScalar ops like dot
///
/// struct MyDenseMatrix<S: LinalgScalar + NumAssign> {
///     matrix: Array2<S>,
/// }
///
/// impl<S: LinalgScalar + NumAssign> LinearOperator<S> for MyDenseMatrix<S> {
///     fn rows(&self) -> usize {
///         self.matrix.nrows()
///     }
///
///     fn cols(&self) -> usize {
///         self.matrix.ncols()
///     }
///
///     fn apply(&self, x: &Array1<S>) -> Array1<S> {
///         self.matrix.dot(x)
///     }
///
///     fn apply_adjoint(&self, x: &Array1<S>) -> Array1<S> {
///         // For a real matrix, adjoint is the transpose.
///         // For complex, it's the conjugate transpose.
///         // LinalgScalar and dot product handle this correctly if S is complex.
///         self.matrix.t().dot(x)
///     }
/// }
/// ```
pub trait LinearOperator<S: LinalgScalar> {
    /// Returns the number of rows in the operator (dimension of the output vector).
    fn rows(&self) -> usize;

    /// Returns the number of columns in the operator (dimension of the input vector).
    fn cols(&self) -> usize;

    /// Applies the operator to a vector `x`, computing `A*x`.
    ///
    /// # Arguments
    ///
    /// * `x`: A reference to an `ndarray::Array1<S>` vector to which the operator
    ///   will be applied. Its dimension must match the operator's column count.
    ///
    /// # Returns
    ///
    /// An `ndarray::Array1<S>` vector resulting from the application `A*x`.
    /// Its dimension will match the operator's row count.
    fn apply(&self, x: &Array1<S>) -> Array1<S>;

    /// Applies the adjoint (Hermitian transpose or transpose) of the operator
    /// to a vector `x`, computing `A^H*x` or `A^T*x`.
    ///
    /// This is required by some Krylov methods (e.g., BiCG, CGS, QMR).
    /// For operators on real vector spaces, this is typically the transpose.
    /// For complex vector spaces, it's the conjugate transpose (Hermitian transpose).
    ///
    /// # Arguments
    ///
    /// * `x`: A reference to an `ndarray::Array1<S>` vector to which the adjoint
    ///   operator will be applied. Its dimension must match the operator's row count.
    ///
    /// # Returns
    ///
    /// An `ndarray::Array1<S>` vector resulting from the application `A^H*x`.
    /// Its dimension will match the operator's column count.
    fn apply_adjoint(&self, x: &Array1<S>) -> Array1<S>;

    /// Returns the shape of the operator as `(rows, cols)`.
    ///
    /// This method provides a convenient way to get both dimensions at once.
    /// It defaults to `(self.rows(), self.cols())`.
    fn shape(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }
}

/// Represents an abstract preconditioner `M`.
///
/// Preconditioners are used to transform the linear system `Ax=b` into an
/// equivalent one that is easier to solve, i.e., has better spectral properties.
/// This often leads to faster convergence of the iterative solver.
///
/// The trait allows for left preconditioning (`M^-1 A x = M^-1 b`) or
/// right preconditioning (`A M^-1 y = b`, where `x = M^-1 y`).
///
/// # Type Parameters
///
/// - `S`: The scalar type, e.g., `f32`, `f64`. Must implement `ndarray::LinalgScalar`.
pub trait Preconditioner<S: LinalgScalar> {
    /// Returns the number of rows in the preconditioner operator.
    /// Typically, for a square preconditioner `M` of size `n x n`, this is `n`.
    fn rows(&self) -> usize;

    /// Returns the number of columns in the preconditioner operator.
    /// Typically, for a square preconditioner `M` of size `n x n`, this is `n`.
    fn cols(&self) -> usize;

    /// Applies the inverse of the preconditioner from the left, computing `M^-1 * x`.
    ///
    /// This is used when the system is transformed to `(M^-1 A) x = M^-1 b`.
    /// The solver then operates on the matrix `M^-1 A`.
    ///
    /// # Arguments
    ///
    /// * `x`: A reference to an `ndarray::Array1<S>` vector.
    ///
    /// # Returns
    ///
    /// An `ndarray::Array1<S>` vector resulting from `M^-1 * x`.
    fn apply_left(&self, x: &Array1<S>) -> Array1<S>;

    /// Applies the inverse of the preconditioner from the right.
    ///
    /// This is typically used to solve `M*y = x` for `y`, effectively computing `y = M^-1 * x`.
    /// When solving `A M^-1 y = b`, the solver operates on `A M^-1`, and the solution
    /// `y` is then used to find `x = M^-1 y`. The vector `x` passed to this function
    /// is the one for which `M*y = x` needs to be solved.
    ///
    /// # Arguments
    ///
    /// * `x`: A reference to an `ndarray::Array1<S>` vector.
    ///
    /// # Returns
    ///
    /// An `ndarray::Array1<S>` vector `y` such that `M*y = x`.
    fn apply_right(&self, x: &Array1<S>) -> Array1<S>;

    /// Returns the shape of the preconditioner as `(rows, cols)`.
    ///
    /// Defaults to `(self.rows(), self.cols())`.
    fn shape(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    // A dummy diagonal operator for testing purposes.
    struct DiagonalOperator {
        diag: Array1<f64>,
    }

    impl DiagonalOperator {
        fn new(diag_elements: Vec<f64>) -> Self {
            Self { diag: Array1::from_vec(diag_elements) }
        }
    }

    impl LinearOperator<f64> for DiagonalOperator {
        fn rows(&self) -> usize {
            self.diag.len()
        }

        fn cols(&self) -> usize {
            self.diag.len()
        }

        fn apply(&self, x: &Array1<f64>) -> Array1<f64> {
            &self.diag * x
        }

        fn apply_adjoint(&self, x: &Array1<f64>) -> Array1<f64> {
            // For a real diagonal matrix, adjoint is the same as apply
            self.apply(x)
        }
    }

    #[test]
    fn test_diagonal_operator_shape_and_rows_cols() {
        let op = DiagonalOperator::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(op.shape(), (3, 3));
        assert_eq!(op.rows(), 3);
        assert_eq!(op.cols(), 3);
    }

    #[test]
    fn test_diagonal_operator_apply() {
        let op = DiagonalOperator::new(vec![1.0, 2.0, 3.0]);
        let x = Array1::from_vec(vec![4.0, 5.0, 6.0]);
        let y = op.apply(&x);
        assert_eq!(y, Array1::from_vec(vec![4.0, 10.0, 18.0]));
    }

     #[test]
    fn test_diagonal_operator_apply_adjoint() {
        let op = DiagonalOperator::new(vec![1.0, 2.0, 3.0]);
        let x = Array1::from_vec(vec![4.0, 5.0, 6.0]);
        let y = op.apply_adjoint(&x);
        assert_eq!(y, Array1::from_vec(vec![4.0, 10.0, 18.0]));
    }

    // A dummy identity preconditioner for testing purposes.
    struct IdentityPreconditioner {
        size: usize,
    }

    impl IdentityPreconditioner {
        fn new(size: usize) -> Self {
            Self { size }
        }
    }

    impl Preconditioner<f64> for IdentityPreconditioner {
        fn rows(&self) -> usize {
            self.size
        }
        fn cols(&self) -> usize {
            self.size
        }

        fn apply_left(&self, x: &Array1<f64>) -> Array1<f64> {
            x.clone()
        }

        fn apply_right(&self, x: &Array1<f64>) -> Array1<f64> {
            x.clone()
        }
    }

    #[test]
    fn test_identity_preconditioner_shape_and_rows_cols() {
        let precond = IdentityPreconditioner::new(4);
        assert_eq!(precond.shape(), (4, 4));
        assert_eq!(precond.rows(), 4);
        assert_eq!(precond.cols(), 4);
    }

    #[test]
    fn test_identity_preconditioner_apply_left() {
        let precond = IdentityPreconditioner::new(3);
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = precond.apply_left(&x);
        assert_eq!(y, x);
    }

    #[test]
    fn test_identity_preconditioner_apply_right() {
        let precond = IdentityPreconditioner::new(3);
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = precond.apply_right(&x);
        assert_eq!(y, x);
    }
}
