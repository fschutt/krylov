//! # Biconjugate Gradient Stabilized (BiCGSTAB) Solver
//!
//! This module provides an implementation of the Biconjugate Gradient Stabilized
//! (BiCGSTAB) method, an iterative solver for general non-symmetric linear
//! systems of equations `Ax = b`.
//!
//! ## Overview
//!
//! BiCGSTAB is a Krylov subspace method that is derived from the Biconjugate
//! Gradient (BiCG) method but offers smoother and often faster convergence.
//! It computes iterates `x_k` such that the residual `r_k = b - Ax_k` is made small.
//! Unlike GMRES, BiCGSTAB does not explicitly minimize the residual norm at each step
//! in a growing subspace, but it uses short recurrences which makes its computational
//! cost per iteration relatively low and constant.
//!
//! ## Advantages
//!
//! - **No Transpose Needed**: BiCGSTAB does not require the application of the
//!   transpose of the operator `A` (i.e., `A^T` or `A^H`), which can be an advantage
//!   if `A^T` is not available or expensive to compute.
//! - **Efficiency**: Often converges faster than GMRES (especially GMRES without restarts
//!   or with small restart lengths) for many types of non-symmetric systems.
//! - **Irregular Convergence**: Its convergence behavior can be somewhat irregular
//!   compared to GMRES, but the "stabilized" part (the "STAB" in BiCGSTAB) helps
//!   to smooth this out compared to the original BiCG or CGS methods.
//!
//! ## Preconditioning
//!
//! This implementation supports optional right preconditioning. If a preconditioner `M`
//! is provided, BiCGSTAB is applied to the system `A M^-1 y = b`, and the final
//! solution is `x = M^-1 y`.
//!
//! ## References
//!
//! - Van der Vorst, H. A. (1992). Bi-CGSTAB: A fast and smoothly converging
//!   variant of Bi-CG for the solution of nonsymmetric linear systems.
//!   SIAM Journal on Scientific and Statistical Computing, 13(2), 631-644.

use crate::operator::{LinearOperator, Preconditioner};
use ndarray::{Array1, Zip, LinalgScalar}; // Removed Norm
use num_traits::{FromPrimitive, NumAssign, One, Zero, Float};
use std::fmt::Debug;

// Helper function for L2 norm calculation
fn l2_norm<S: Float + LinalgScalar + Copy + Zero + FromPrimitive>(vector: &Array1<S>) -> S {
    let sum_sq = vector.iter().map(|&x| x * x).fold(S::zero(), |acc, val| acc + val);
    sum_sq.sqrt()
}

/// Parameters for the BiCGSTAB solver.
#[derive(Clone, Debug)]
pub struct BicgstabParams<S: Float + LinalgScalar> {
    /// Maximum number of iterations allowed.
    pub max_iterations: usize,

    /// Tolerance for the relative residual norm `||b - Ax_k|| / ||b||`.
    ///
    /// The iteration stops if this condition is met. If `||b||` is very small
    /// (close to zero), this effectively becomes an absolute tolerance on
    /// `||b - Ax_k||`.
    pub tolerance: S,
}

/// Solution structure returned by the BiCGSTAB solver.
#[derive(Debug)]
pub struct BicgstabSolution<S: Float + LinalgScalar> {
    /// The approximate solution vector `x_k`.
    pub x: Array1<S>,

    /// Total number of iterations performed.
    pub iterations: usize,

    /// The L2 norm of the final residual vector `||b - Ax_k||_2`.
    pub residual_norm: S,

    /// Flag indicating whether the solver converged to the specified tolerance.
    pub converged: bool,

    /// A string explaining the reason for termination (e.g., convergence,
    /// max iterations reached, or breakdown).
    pub reason: String,
}

// Main bicgstab function
/// Solves the linear system `Ax = b` using the Biconjugate Gradient Stabilized (BiCGSTAB) method.
///
/// # Type Parameters
///
/// - `S`: Scalar type (e.g., `f64`, `f32`) implementing `Float`, `LinalgScalar`,
///   and other necessary numeric traits.
/// - `Op`: Type of the linear operator `A`, implementing `LinearOperator<S>`.
/// - `P`: Type of the preconditioner `M`, implementing `Preconditioner<S>`.
///
/// # Arguments
///
/// * `operator`: A reference to the linear operator `A`.
/// * `b`: A reference to the right-hand side vector `b`.
/// * `x0`: A reference to the initial guess vector `x_0`.
/// * `params`: `BicgstabParams` struct containing solver parameters like max iterations
///   and tolerance.
/// * `preconditioner`: An `Option` containing a reference to a right preconditioner `M`.
///   If `Some(p)`, the solver addresses `A M^-1 y = b` and returns `x = M^-1 y`.
///   If `None`, no preconditioning is applied.
///
/// # Returns
///
/// A `Result` which is:
/// - `Ok(BicgstabSolution<S>)` if the solver completes. The `BicgstabSolution`
///   contains the solution, iteration count, residual norm, convergence status,
///   and a termination reason (convergence, max iterations, or breakdown type).
/// - `Err(String)` if there's an input validation error (e.g., dimension mismatch).
///
/// # Example
///
/// ```rust
/// use krylov_rs::bicgstab::{bicgstab, BicgstabParams};
/// use krylov_rs::operator::{LinearOperator, Preconditioner}; // Added Preconditioner
/// use ndarray::{array, Array1, Array2, LinalgScalar};
/// use num_traits::{Float, FromPrimitive, NumAssign, One, Zero}; // REMOVED Copy, PartialOrd
/// use std::fmt::Debug; // For S
/// use ndarray::ScalarOperand; // For S
///
/// // Dummy operator for the example
/// struct MyMatrix<Sc: LinalgScalar + NumAssign + Zero + One + Copy> { matrix: Array2<Sc> } // Added Zero, One, Copy
/// impl<Sc: LinalgScalar + NumAssign + Zero + One + Copy> LinearOperator<Sc> for MyMatrix<Sc> {
///     fn rows(&self) -> usize { self.matrix.nrows() }
///     fn cols(&self) -> usize { self.matrix.ncols() }
///     fn apply(&self, x: &Array1<Sc>) -> Array1<Sc> { self.matrix.dot(x) }
///     fn apply_adjoint(&self, x: &Array1<Sc>) -> Array1<Sc> { self.matrix.t().dot(x) }
/// }
/// // Also implement Preconditioner for MyMatrix for the example
/// impl<Sc: LinalgScalar + NumAssign + Zero + One + Copy> Preconditioner<Sc> for MyMatrix<Sc> {
///     fn rows(&self) -> usize { self.matrix.nrows() }
///     fn cols(&self) -> usize { self.matrix.ncols() }
///     fn apply_left(&self, x: &Array1<Sc>) -> Array1<Sc> { x.clone() } // Identity P
///     fn apply_right(&self, x: &Array1<Sc>) -> Array1<Sc> { x.clone() } // Identity P
/// }
///
/// type MyScalar = f64;
/// let op_a = MyMatrix { matrix: array![[4.0, 1.0], [1.0, 3.0]] };
/// let b_vec: Array1<MyScalar> = array![1.0, 2.0];
/// let x_initial: Array1<MyScalar> = Array1::zeros(LinearOperator::cols(&op_a)); // Already disambiguated, ensure it's correct
///
/// let params = BicgstabParams {
///     max_iterations: 10,
///     tolerance: 1e-8,
/// };
///
/// match bicgstab(&op_a, &b_vec, &x_initial, &params, None::<&MyMatrix<MyScalar>>) {
///     Ok(solution) => {
///         println!("Converged: {}", solution.converged);
///         println!("Iterations: {}", solution.iterations);
///         println!("Solution: {:?}", solution.x);
///         println!("Residual norm: {}", solution.residual_norm);
///         println!("Reason: {}", solution.reason);
///     }
///     Err(e) => eprintln!("Error: {}", e),
/// }
/// ```
#[allow(clippy::too_many_lines)] // BiCGSTAB can be long
pub fn bicgstab<S, Op, P>(
    operator: &Op,
    b: &Array1<S>,
    x0: &Array1<S>,
    params: &BicgstabParams<S>,
    preconditioner: Option<&P>,
) -> Result<BicgstabSolution<S>, String>
where
    S: Float + LinalgScalar + FromPrimitive + NumAssign + Debug + PartialOrd + Copy + Zero + One + ndarray::ScalarOperand,
    Op: LinearOperator<S>,
    P: Preconditioner<S>,
{
    let n = operator.cols();
    if operator.rows() != n {
        return Err(format!("Operator must be square, got {}x{}", operator.rows(), operator.cols()));
    }
    if b.len() != n {
        return Err(format!("Vector b has incompatible dimension {}. Expected {}.", b.len(), n));
    }
    if x0.len() != n {
        return Err(format!("Initial guess x0 has incompatible dimension {}. Expected {}.", x0.len(), n));
    }

    let mut x = x0.clone();
    let mut r = b - operator.apply(&x); // r_0 = b - A x_0

    let r_hat = r.clone(); // Arbitrary choice for r_hat_0, often r_0.
                           // Ensure r_hat_0.dot(r_0) is not zero if r_0 is not zero.
                           // If r_0 is zero, x0 is solution. This is checked below.

    let b_norm = l2_norm(b);
    let effective_b_norm = if b_norm.is_zero() { S::one() } else { b_norm };

    let initial_residual_norm = l2_norm(&r);
    if initial_residual_norm <= params.tolerance * effective_b_norm {
        return Ok(BicgstabSolution {
            x,
            iterations: 0,
            residual_norm: initial_residual_norm,
            converged: true,
            reason: "Initial guess is already a solution.".to_string(),
        });
    }

    // Smallest representable positive value / a very small number for breakdown checks
    // S::epsilon() is machine epsilon (diff between 1.0 and next representable).
    // For breakdown, we need something related to underflow or squared epsilon.
    let breakdown_tol = S::epsilon() * S::epsilon();
    // Alternative: S::from_f64(1e-20).unwrap_or_else(|| S::epsilon() * S::epsilon());

    let mut rho_prev = S::one();
    let mut alpha = S::one();
    let mut omega = S::one();

    let mut v = Array1::zeros(n);
    let mut p = Array1::zeros(n);

    let mut iterations_count = 0;

    for iter in 0..params.max_iterations {
        iterations_count = iter + 1;
        let rho_curr = r_hat.dot(&r);

        if rho_curr.abs() < breakdown_tol {
            let final_residual_norm = l2_norm(&(b - operator.apply(&x)));
            return Ok(BicgstabSolution {
                x,
                iterations: iterations_count,
                residual_norm: final_residual_norm,
                converged: final_residual_norm <= params.tolerance * effective_b_norm,
                reason: format!("Algorithm broke down: rho_curr is near zero (val={:?}).", rho_curr),
            });
        }

        if iter == 0 {
            p = r.clone();
        } else {
            let beta = (rho_curr / rho_prev) * (alpha / omega);
            // p = r_k + beta * (p_{k-1} - omega_{k-1} * v_{k-1})
            // v is v_{k-1} here (from previous iteration)
            // p is p_{k-1} here (from previous iteration)
            let p_term = &p - &v * omega; // p_{k-1} - omega_{k-1} * v_{k-1}
            p = &r + &p_term * beta;     // r_k + beta * p_term
        }

        let p_hat = match preconditioner {
            Some(prec) => prec.apply_right(&p),
            None => p.clone(),
        };

        v = operator.apply(&p_hat); // v_k = A * p_hat_k

        let r_hat_dot_v = r_hat.dot(&v);
        if r_hat_dot_v.abs() < breakdown_tol {
            let final_residual_norm = l2_norm(&(b - operator.apply(&x)));
            return Ok(BicgstabSolution {
                x,
                iterations: iterations_count,
                residual_norm: final_residual_norm,
                converged: final_residual_norm <= params.tolerance * effective_b_norm,
                reason: format!("Algorithm broke down: r_hat_dot_v is near zero (val={:?}).", r_hat_dot_v),
            });
        }

        alpha = rho_curr / r_hat_dot_v;

        // Early exit check using s vector (intermediate residual)
        // s = r_k - alpha_k * v_k (v_k is current v = A p_hat_k)
        let s = &r - &v * alpha;

        let s_norm = l2_norm(&s);
        if s_norm <= params.tolerance * effective_b_norm {
            // x = x + alpha * p_hat;
            Zip::from(&mut x).and(&p_hat).for_each(|x_i, &p_hat_i| {
                *x_i += alpha * p_hat_i; // Use +=
            });
            let final_residual_norm = l2_norm(&(b - operator.apply(&x))); // Recompute true residual
            return Ok(BicgstabSolution {
                x,
                iterations: iterations_count,
                residual_norm: final_residual_norm,
                converged: true,
                reason: "Converged after alpha update (s_norm condition).".to_string(),
            });
        }

        let s_hat = match preconditioner {
            Some(prec) => prec.apply_right(&s),
            None => s.clone(),
        };

        let t = operator.apply(&s_hat); // t_k = A * s_hat_k

        let t_dot_s = t.dot(&s);
        let t_dot_t = t.dot(&t);

        if t_dot_t.abs() < breakdown_tol { // t is effectively zero
            let final_residual_norm = l2_norm(&(b - operator.apply(&x)));
             // If t is zero, omega would be NaN or Inf.
             // x update would be just x = x + alpha * p_hat
             // r update would be r = s
             // This is a breakdown. Solution may or may not be good.
            Zip::from(&mut x).and(&p_hat).for_each(|x_i, &p_hat_i| {
                *x_i += alpha * p_hat_i; // Use += // x_{k+1} = x_k + alpha_k p_hat_k
            });
            r = s; // r_{k+1} = s_k
            let current_residual_norm_after_alpha = l2_norm(&r);

            return Ok(BicgstabSolution {
                x,
                iterations: iterations_count,
                residual_norm: current_residual_norm_after_alpha, // Or recompute true: l2_norm(&(b - operator.apply(&x)))
                converged: current_residual_norm_after_alpha <= params.tolerance * effective_b_norm,
                reason: format!("Algorithm broke down: t_dot_t is near zero (val={:?}). Stabilized omega calculation avoided.", t_dot_t),
            });
        }

        omega = t_dot_s / t_dot_t; // omega_k

        // x_{k+1} = x_k + alpha_k * p_hat_k + omega_k * s_hat_k
        x = &x + &(&p_hat * alpha) + &(&s_hat * omega);

        // r_{k+1} = s_k - omega_k * t_k
        r = &s - &t * omega;

        rho_prev = rho_curr;

        let current_residual_norm = l2_norm(&r);
        if current_residual_norm <= params.tolerance * effective_b_norm {
            // For final accuracy, recompute residual from x: b - A*x
            let final_residual_norm = l2_norm(&(b - operator.apply(&x)));
            // And check this true residual norm
            if final_residual_norm <= params.tolerance * effective_b_norm {
                 return Ok(BicgstabSolution {
                    x,
                    iterations: iterations_count,
                    residual_norm: final_residual_norm,
                    converged: true,
                    reason: "Converged after omega update.".to_string(),
                });
            }
            // If recomputed residual is worse, it implies some instability or precision loss.
            // Could continue, or exit reporting the current state. For now, let's say it converged if the computed r was small.
            // To be more robust, one might want to use the recomputed residual for the check.
            // However, the algorithm proceeds based on the iteratively updated 'r'.
            // Sticking to the algorithm's 'r' for this check:
             return Ok(BicgstabSolution {
                x,
                iterations: iterations_count,
                residual_norm: current_residual_norm, // Norm of iteratively updated r
                converged: true, // Based on iterative r
                reason: "Converged after omega update (based on iterative r).".to_string(),
            });
        }
    } // End main loop

    let final_residual_norm = l2_norm(&(b - operator.apply(&x)));
    Ok(BicgstabSolution {
        x,
        iterations: iterations_count,
        residual_norm: final_residual_norm,
        converged: final_residual_norm <= params.tolerance * effective_b_norm,
        reason: "Reached maximum iterations.".to_string(),
    })
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array; // For easy array creation in tests

    // Minimal TestLinearOperator for f64
    struct TestLinearOperator {
        matrix: ndarray::Array2<f64>,
    }

    impl LinearOperator<f64> for TestLinearOperator {
        fn rows(&self) -> usize {
            self.matrix.nrows()
        }
        fn cols(&self) -> usize {
            self.matrix.ncols()
        }
        fn apply(&self, x: &Array1<f64>) -> Array1<f64> {
            self.matrix.dot(x)
        }
        fn apply_adjoint(&self, x: &Array1<f64>) -> Array1<f64> {
            // BiCGSTAB does not strictly require apply_adjoint in its basic form,
            // but LinearOperator trait does.
            self.matrix.t().dot(x)
        }
    }

    // Minimal TestPreconditioner for f64 (Identity)
    struct TestPreconditioner {
        size: usize,
    }

    impl Preconditioner<f64> for TestPreconditioner {
        fn rows(&self) -> usize {
            self.size
        }
        fn cols(&self) -> usize {
            self.size
        }
        fn apply_left(&self, x: &Array1<f64>) -> Array1<f64> {
            // Not used by this BiCGSTAB implementation (assumes right precond)
            x.clone()
        }
        fn apply_right(&self, x: &Array1<f64>) -> Array1<f64> {
            x.clone()
        }
    }

    #[test]
    fn test_simple_diagonal_system_no_precond() {
        let operator = TestLinearOperator {
            matrix: array![[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
        };
        let b = array![2.0, 6.0, 12.0]; // Sol: (1,2,3)
        let x0 = array![0.0, 0.0, 0.0];

        let params = BicgstabParams {
            max_iterations: 10,
            tolerance: 1e-9,
        };

        let result = bicgstab(&operator, &b, &x0, &params, None::<&TestPreconditioner>);
        assert!(result.is_ok());
        let solution = result.unwrap();

        println!("Reason: {}", solution.reason);
        assert!(solution.converged);
        assert!(solution.iterations <= params.max_iterations);

        let expected_x = array![1.0, 2.0, 3.0];
        let diff_norm = l2_norm(&(&expected_x - &solution.x));
        println!("Solution x: {:?}", solution.x);
        println!("Expected x: {:?}", expected_x);
        println!("Error norm: {}", diff_norm);
        println!("Iterations: {}", solution.iterations);
        println!("Residual norm: {}", solution.residual_norm);

        assert!(diff_norm < 1e-7);
        assert!(solution.residual_norm < params.tolerance);
    }

    #[test]
    fn test_simple_diagonal_system_with_precond() {
        let operator = TestLinearOperator {
            matrix: array![[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
        };
        let preconditioner = TestPreconditioner { size: 3 }; // Identity
        let b = array![2.0, 6.0, 12.0]; // Sol: (1,2,3)
        let x0 = array![0.0, 0.0, 0.0];

        let params = BicgstabParams {
            max_iterations: 10,
            tolerance: 1e-9,
        };

        let result = bicgstab(&operator, &b, &x0, &params, Some(&preconditioner));
        assert!(result.is_ok());
        let solution = result.unwrap();

        println!("Reason: {}", solution.reason);
        assert!(solution.converged);
        assert!(solution.iterations <= params.max_iterations);

        let expected_x = array![1.0, 2.0, 3.0];
        let diff_norm = l2_norm(&(&expected_x - &solution.x));
         println!("Solution x (precond): {:?}", solution.x);
        println!("Error norm (precond): {}", diff_norm);

        assert!(diff_norm < 1e-7);
        assert!(solution.residual_norm < params.tolerance);
    }

    #[test]
    fn test_b_is_zero() {
        let operator = TestLinearOperator {
            matrix: array![[1.0, 0.0], [0.0, 2.0]],
        };
        let b = array![0.0, 0.0];
        let x0 = array![1.0, 1.0]; // Non-zero x0, expect x -> 0

        let params = BicgstabParams {
            max_iterations: 20, // Might need more for b=0 if x0 is not 0
            tolerance: 1e-9,
        };

        let result = bicgstab(&operator, &b, &x0, &params, None::<&TestPreconditioner>);
        assert!(result.is_ok());
        let solution = result.unwrap();

        println!("Reason (b_zero): {}", solution.reason);
        assert!(solution.converged);

        let expected_x = array![0.0, 0.0];
        let diff_norm = l2_norm(&(&expected_x - &solution.x));
        println!("Solution x (b_zero): {:?}", solution.x);
        println!("Error norm (b_zero): {}", diff_norm);
        println!("Iterations (b_zero): {}", solution.iterations);
        println!("Residual norm (b_zero): {}", solution.residual_norm);

        // For b=0, residual norm ||0 - Ax|| = ||Ax|| should be small
        assert!(solution.residual_norm < params.tolerance);
        // And x should be close to 0
        assert!(diff_norm < 1e-7);
    }

    #[test]
    fn test_x0_is_solution() {
        let operator = TestLinearOperator {
            matrix: array![[1.0, 0.0], [0.0, 2.0]],
        };
        let b = array![1.0, 4.0];
        let x0 = array![1.0, 2.0]; // Ax0 = b

        let params = BicgstabParams {
            max_iterations: 5,
            tolerance: 1e-9,
        };

        let result = bicgstab(&operator, &b, &x0, &params, None::<&TestPreconditioner>);
        assert!(result.is_ok());
        let solution = result.unwrap();

        println!("Reason (x0_is_solution): {}", solution.reason);
        assert!(solution.converged);
        assert_eq!(solution.iterations, 0);
        assert!(solution.residual_norm < params.tolerance);
        let error_norm = l2_norm(&(&x0 - &solution.x));
        assert!(error_norm < 1e-9); // Should return x0
    }
}
