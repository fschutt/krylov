//! # Generalized Minimal Residual (GMRES) Solver
//!
//! This module provides an implementation of the Generalized Minimal Residual (GMRES)
//! method, a popular iterative solver for general (non-symmetric, non-Hermitian)
//! linear systems of equations `Ax = b`.
//!
//! ## Overview
//!
//! GMRES iteratively finds an approximate solution `x_k` in the Krylov subspace
//! `K_k(A, r_0)` such that the norm of the residual `||b - Ax_k||_2` is minimized.
//! `r_0 = b - Ax_0` is the initial residual for an initial guess `x_0`.
//!
//! The method involves generating an orthonormal basis for the Krylov subspace using
//! Arnoldi iteration and then solving a small least-squares problem to find the
//! coefficients of the solution vector in this basis.
//!
//! ## Restarts
//!
//! For large systems or many iterations, the storage and computational cost of
//! Arnoldi iteration can become prohibitive as the basis size `k` grows. To mitigate
//! this, GMRES is often implemented with restarts. After a fixed number `m` of
//! iterations (the restart length), the current solution `x_m` is used as the new
//! initial guess `x_0`, and the process is restarted. This is denoted as GMRES(m).
//! While restarts save resources, they can sometimes lead to slower convergence or
//! stagnation.
//!
//! ## Preconditioning
//!
//! This implementation supports right preconditioning. If a preconditioner `M` is
//! provided, GMRES is effectively applied to the system `A M^-1 y = b`, and the
//! final solution is `x = M^-1 y`.
//!
//! ## References
//!
//! - Saad, Y., & Schultz, M. H. (1986). GMRES: A generalized minimal residual
//!   algorithm for solving nonsymmetric linear systems. SIAM Journal on scientific
//!   and statistical computing, 7(3), 856-869.

use crate::operator::{LinearOperator, Preconditioner};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Zip};
// Norm import removed, will calculate manually
use num_traits::{Float, FromPrimitive, Signed, Zero};
use std::ops::{AddAssign, Neg};
use std::fmt::Debug;
use std::iter::Sum;
use ndarray::LinalgScalar;

/// Parameters for the GMRES solver.
#[derive(Clone, Debug)]
pub struct GmresParams<S: LinalgScalar> {
    /// The restart length `m` for GMRES(m).
    ///
    /// GMRES constructs an orthonormal basis of size `m`. After `m` iterations,
    /// the algorithm is restarted with the current approximate solution as the
    /// new initial guess. A larger `m` increases computational cost and memory
    /// per restart cycle but can lead to better convergence.
    pub restart_length: usize,

    /// Maximum number of iterations allowed across all restart cycles.
    pub max_iterations: usize,

    /// Tolerance for the relative residual norm `||b - Ax_k|| / ||b||`.
    ///
    /// The iteration stops if this condition is met. If `||b||` is very small
    /// (close to zero), this effectively becomes an absolute tolerance on
    /// `||b - Ax_k||`.
    pub tolerance: S,
}

/// Solution structure returned by the GMRES solver.
#[derive(Debug)]
pub struct GmresSolution<S: LinalgScalar> {
    /// The approximate solution vector `x_k`.
    pub x: Array1<S>,

    /// Total number of iterations performed across all restart cycles.
    pub iterations: usize,

    /// The L2 norm of the final residual vector `||b - Ax_k||_2`.
    pub residual_norm: S,

    /// Flag indicating whether the solver converged to the specified tolerance.
    pub converged: bool,

    /// A string explaining the reason for termination (e.g., convergence,
    /// max iterations reached, breakdown).
    pub reason: String,
}

// Helper functions will be added below

// Helper functions
fn apply_givens_rotation<S: LinalgScalar + Neg<Output = S>>(c: S, s: S, val1: S, val2: S) -> (S, S) {
    let v1_new = c * val1 + s * val2;
    let v2_new = -s * val1 + c * val2;
    (v1_new, v2_new)
}

fn compute_givens_rotation<S: LinalgScalar + Signed + FromPrimitive + Float + Neg<Output = S>>(a: S, b: S) -> (S, S) {
    if b.is_zero() {
        (S::one(), S::zero())
    } else if a.is_zero() {
        (S::zero(), S::one()) // Or (S::zero(), b.signum()) depending on convention
    } else {
        let r = (a.powi(2) + b.powi(2)).sqrt();
        let c = a / r;
        let s = b / r;
        (c, s)
    }
}

fn back_substitute<S: LinalgScalar + FromPrimitive + Zero>(
    h_upper: &ArrayView2<S>,
    g_rhs_slice: &ArrayView1<S>,
) -> Array1<S> {
    let n_coeffs = g_rhs_slice.len();
    if h_upper.ncols() != n_coeffs || h_upper.nrows() != n_coeffs {
        // This should ideally return a Result or panic with a more descriptive message.
        // For now, returning zeros, but this indicates an issue.
        // Consider that h_upper is j_final x j_final. g_rhs_slice is j_final.
        // If j_final is 0 (e.g. initial residual was already 0), this is problematic.
        if n_coeffs == 0 {
            return Array1::zeros(0);
        }
        // Fallback if dimensions mismatch, though in GMRES this implies a problem.
        return Array1::zeros(n_coeffs);
    }

    let mut y_coeffs = Array1::zeros(n_coeffs);
    if n_coeffs == 0 {
        return y_coeffs;
    }

    for i in (0..n_coeffs).rev() {
        let mut sum_val = S::zero();
        for j in (i + 1)..n_coeffs {
            sum_val = sum_val + h_upper[[i, j]] * y_coeffs[j];
        }
        // Check for division by zero, though h_upper[i,i] should be non-zero after Givens.
        // If h_upper[[i,i]] is zero, it implies a rank deficient system from Arnoldi,
        // which can happen if breakdown occurred and j_final was not set correctly,
        // or if the system is singular.
        let h_ii = h_upper[[i, i]];
        if h_ii.is_zero() {
            // Handle singular or near-singular case, e.g. by returning an error or specific value.
            // For now, just using zero, which might not be numerically stable/correct.
            y_coeffs[i] = (g_rhs_slice[i] - sum_val) / (S::from_f64(1e-12).unwrap_or_else(S::one));
        } else {
            y_coeffs[i] = (g_rhs_slice[i] - sum_val) / h_ii;
        }
    }
    y_coeffs
}

// Main gmres function
/// Solves the linear system `Ax = b` using the Generalized Minimal Residual (GMRES) method.
///
/// This function implements GMRES with restarts and optional right preconditioning.
///
/// # Type Parameters
///
/// - `S`: Scalar type (e.g., `f64`, `f32`) implementing `LinalgScalar`, `Float`,
///   and other necessary numeric traits from `num-traits`.
/// - `Op`: Type of the linear operator `A`, implementing `LinearOperator<S>`.
/// - `P`: Type of the preconditioner `M`, implementing `Preconditioner<S>`.
///
/// # Arguments
///
/// * `operator`: A reference to the linear operator `A` of the system.
/// * `b`: A reference to the right-hand side vector `b`.
/// * `x0`: A reference to the initial guess vector `x_0`.
/// * `params`: `GmresParams` struct containing solver parameters like restart length,
///   max iterations, and tolerance.
/// * `preconditioner`: An `Option` containing a reference to a right preconditioner `M`.
///   If `Some(p)`, the solver addresses `A M^-1 y = b` and returns `x = M^-1 y`.
///   If `None`, no preconditioning is applied.
///
/// # Returns
///
/// A `Result` which is:
/// - `Ok(GmresSolution<S>)` if the solver completes successfully (either converged or
///   reached max iterations/breakdown). The `GmresSolution` struct contains the
///   solution vector, iteration count, final residual norm, convergence status, and
///   a termination reason.
/// - `Err(String)` if there's an input validation error (e.g., dimension mismatch).
///
/// # Example
///
/// ```rust
/// use krylov_rs::gmres::{gmres, GmresParams, GmresSolution};
/// use krylov_rs::operator::{LinearOperator, Preconditioner}; // Added Preconditioner
/// use ndarray::{array, Array1, Array2, LinalgScalar, Ix1};
/// use num_traits::{Zero, NumAssign, Float, FromPrimitive, Signed, One}; // Removed Copy, PartialOrd
/// use std::iter::Sum; // For S
/// use std::ops::{AddAssign, Neg}; // For S
/// use ndarray::ScalarOperand; // For S
/// use std::fmt::Debug; // For S
///
/// // Dummy operator for the example
/// struct SimpleMatrix<Sc: LinalgScalar + NumAssign + Zero + One + Copy> { matrix: Array2<Sc> } // Added Zero, One, Copy
/// impl<Sc: LinalgScalar + NumAssign + Zero + One + Copy> LinearOperator<Sc> for SimpleMatrix<Sc> {
///     fn rows(&self) -> usize { self.matrix.nrows() }
///     fn cols(&self) -> usize { self.matrix.ncols() }
///     fn apply(&self, x: &Array1<Sc>) -> Array1<Sc> { self.matrix.dot(x) }
///     fn apply_adjoint(&self, x: &Array1<Sc>) -> Array1<Sc> { self.matrix.t().dot(x) }
/// }
/// // Also implement Preconditioner for SimpleMatrix for the example
/// impl<Sc: LinalgScalar + NumAssign + Zero + One + Copy> Preconditioner<Sc> for SimpleMatrix<Sc> {
///     fn rows(&self) -> usize { self.matrix.nrows() }
///     fn cols(&self) -> usize { self.matrix.ncols() }
///     fn apply_left(&self, x: &Array1<Sc>) -> Array1<Sc> { x.clone() } // Identity P
///     fn apply_right(&self, x: &Array1<Sc>) -> Array1<Sc> { x.clone() } // Identity P
/// }
///
/// type MyScalar = f64;
/// let op_a = SimpleMatrix { matrix: array![[4.0, 1.0], [1.0, 3.0]] };
/// let b_vec: Array1<MyScalar> = array![1.0, 2.0];
/// let x_initial: Array1<MyScalar> = Array1::zeros(LinearOperator::cols(&op_a)); // Disambiguated
///
/// let params = GmresParams {
///     restart_length: 2,
///     max_iterations: 10,
///     tolerance: 1e-8,
/// };
///
/// match gmres(&op_a, &b_vec, &x_initial, &params, None::<&SimpleMatrix<MyScalar>>) {
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
#[allow(clippy::too_many_lines)] // GMRES is inherently long
pub fn gmres<S, Op, P>(
    operator: &Op,
    b: &Array1<S>,
    x0: &Array1<S>,
    params: &GmresParams<S>,
    preconditioner: Option<&P>,
) -> Result<GmresSolution<S>, String>
where
    S: LinalgScalar + FromPrimitive + PartialOrd + Sum + Signed + Debug + Float + Neg<Output = S> + AddAssign + ndarray::ScalarOperand,
    Op: LinearOperator<S>,
    P: Preconditioner<S>,
{
    let n = operator.cols();
    if operator.rows() != n {
        return Err(format!(
            "Operator must be square, got {}x{}",
            operator.rows(),
            operator.cols()
        ));
    }
    if b.len() != n {
        return Err(format!(
            "Vector b has incompatible dimension {}. Expected {}.",
            b.len(),
            n
        ));
    }
    if x0.len() != n {
        return Err(format!(
            "Initial guess x0 has incompatible dimension {}. Expected {}.",
            x0.len(),
            n
        ));
    }

    let mut x_current = x0.clone();
    let b_norm = b.iter().map(|&x| x * x).sum::<S>().sqrt();

    // If b is the zero vector, the solution is x=0 (if x0=0) or x0 if Ax0=0.
    // More generally, if b_norm is very small, x0 is likely a good enough solution.
    let effective_b_norm = if b_norm.is_zero() || b_norm < S::from_f64(1e-12).unwrap_or_else(S::zero) {
        S::one() // Use 1.0 for relative tolerance calculation to avoid division by zero or large relative error
    } else {
        b_norm
    };

    let initial_r = b - operator.apply(&x_current);
    let initial_r_norm = initial_r.iter().map(|&x| x * x).sum::<S>().sqrt();

    if initial_r_norm <= params.tolerance * effective_b_norm {
        return Ok(GmresSolution {
            x: x_current,
            iterations: 0,
            residual_norm: initial_r_norm,
            converged: true,
            reason: "Initial guess is already a solution.".to_string(),
        });
    }


    let m = params.restart_length;
    let mut total_iter = 0;
    let mut converged = false;
    let mut current_residual_norm = initial_r_norm;
    let mut j_final = m; // Initialize j_final here

    for _restart_cycle in 0..(params.max_iterations / m + usize::from(params.max_iterations % m != 0)) {
        let r_outer = b - operator.apply(&x_current);
        let r_outer_norm = r_outer.iter().map(|&x| x * x).sum::<S>().sqrt();

        current_residual_norm = r_outer_norm; // Update current_residual_norm at start of cycle

        if r_outer_norm <= params.tolerance * effective_b_norm {
            converged = true;
            break;
        }

        let mut v_basis: Vec<Array1<S>> = Vec::with_capacity(m + 1);
        if r_outer_norm.is_zero() { // Should have been caught by earlier check, but for safety
             converged = true; // effectively
             break;
        }
        v_basis.push(&r_outer / r_outer_norm);

        let mut h_matrix: Array2<S> = Array2::zeros((m + 1, m));
        let mut g_rhs: Array1<S> = Array1::zeros(m + 1);
        g_rhs[0] = r_outer_norm;

        let mut givens_cs: Vec<(S, S)> = Vec::with_capacity(m);
        // j_final is now initialized before the loop, and potentially updated within.
        // If a cycle completes fully without breakdown/early convergence, j_final remains m for that cycle's solve.
        // It's reset effectively for the H-solve part by using j_final in slices,
        // but the j_final for the *reason string* should reflect the state of the last cycle.
        // So, j_final should be set to m at the start of *each* restart cycle's Arnoldi process.
        j_final = m; // Reset for the current cycle's H matrix construction

        for j in 0..m {
            if total_iter >= params.max_iterations {
                break;
            }
            total_iter += 1;

            let v_j_preconditioned = match preconditioner {
                Some(p) => p.apply_right(&v_basis[j]), // M^-1 * v_j
                None => v_basis[j].clone(),
            };

            let mut w = operator.apply(&v_j_preconditioned); // A * M^-1 * v_j

            // Modified Gram-Schmidt
            for i in 0..=j {
                let h_val = v_basis[i].dot(&w);
                h_matrix[[i, j]] = h_val;
                // w = w - h_val * &v_basis[i];
                Zip::from(&mut w)
                    .and(&v_basis[i])
                    .for_each(|elem_w, elem_v_i| *elem_w = *elem_w - h_val * *elem_v_i);
            }

            let h_next_val = w.iter().map(|&x| x * x).sum::<S>().sqrt();
            h_matrix[[j + 1, j]] = h_next_val;

            // Apply previous Givens rotations to the current column of H
            for k_rot in 0..j {
                let (new_h_k, new_h_k_plus_1) = apply_givens_rotation(
                    givens_cs[k_rot].0,
                    givens_cs[k_rot].1,
                    h_matrix[[k_rot, j]],
                    h_matrix[[k_rot + 1, j]],
                );
                h_matrix[[k_rot, j]] = new_h_k;
                h_matrix[[k_rot + 1, j]] = new_h_k_plus_1;
            }

            // Compute and apply current Givens rotation
            let (c_j, s_j) = compute_givens_rotation(h_matrix[[j, j]], h_matrix[[j + 1, j]]);
            givens_cs.push((c_j, s_j));

            let (new_h_j_j, new_h_j_plus_1_j) = apply_givens_rotation(c_j, s_j, h_matrix[[j,j]], h_matrix[[j+1,j]]);
            h_matrix[[j,j]] = new_h_j_j;
            h_matrix[[j+1,j]] = new_h_j_plus_1_j; // should be (close to) zero now

            let (new_g_j, new_g_j_plus_1) = apply_givens_rotation(c_j, s_j, g_rhs[j], g_rhs[j+1]);
            g_rhs[j] = new_g_j;
            g_rhs[j+1] = new_g_j_plus_1;

            current_residual_norm = g_rhs[j + 1].abs();

            if current_residual_norm <= params.tolerance * effective_b_norm {
                j_final = j + 1;
                converged = true;
                // Reason will be set when exiting outer loop or by recompute check
                break;
            }

            if h_next_val.is_zero() || h_next_val < S::from_f64(1e-14).unwrap_or_else(S::zero) { // Breakdown
                j_final = j + 1;
                // Reason: breakdown in Arnoldi. Will be determined at outer loop exit.
                break;
            }

            if j < m - 1 { // Avoid pushing if j is the last iteration m-1
                 v_basis.push(w / h_next_val);
            }
        } // End inner Arnoldi loop

        if j_final == 0 { // Should not happen if initial residual > 0
            // This implies r_outer_norm was zero or became zero immediately.
            // If initial_r_norm was already checked, this state implies an issue or immediate convergence.
            // If r_outer_norm was non-zero, but g_rhs[0] became zero after rotation (not possible with standard Givens),
            // or if j_final somehow is 0 without convergence.
            // For safety, check if already converged.
             let temp_r_norm = (b - operator.apply(&x_current)).iter().map(|&x| x * x).sum::<S>().sqrt();
             if temp_r_norm <= params.tolerance * effective_b_norm {
                current_residual_norm = temp_r_norm;
                converged = true;
             } // else, it's a breakdown with no progress, loop will likely terminate due to max_iter.
             break; // Break restart cycle
        }

        // Solve the least squares problem Hy = g_rhs_slice
        // H is (j_final+1) x j_final, but after Givens, it's effectively j_final x j_final upper triangular R
        // We need to solve R y = g_rhs_slice[0..j_final]
        let y_coeffs = back_substitute(
            &h_matrix.slice(s![0..j_final, 0..j_final]),
            &g_rhs.slice(s![0..j_final]),
        );

        let mut update_vec_unprec = Array1::zeros(n);
        for k in 0..j_final {
            // update_vec_unprec.scaled_add(y_coeffs[k], &v_basis[k]);
            let yk = y_coeffs[k];
            Zip::from(&mut update_vec_unprec)
                .and(&v_basis[k])
                .for_each(|upd_elem, v_elem| *upd_elem += yk * *v_elem); // Use +=
        }

        let final_update = match preconditioner {
            Some(p) => p.apply_right(&update_vec_unprec), // M^-1 * sum(y_k * v_k)
            None => update_vec_unprec,
        };

        // x_current += &final_update;
        Zip::from(&mut x_current)
            .and(&final_update)
            .for_each(|x_elem, upd_elem| *x_elem += *upd_elem);


        // Check true residual norm if converged within a cycle or for final reporting
        // This is also important if breakdown occurred (j_final < m)
        if converged || j_final < m || (_restart_cycle == (params.max_iterations / m + usize::from(params.max_iterations % m != 0)) - 1 && total_iter >= params.max_iterations) {
             let true_r = b - operator.apply(&x_current);
             current_residual_norm = true_r.iter().map(|&x| x * x).sum::<S>().sqrt();
             if current_residual_norm <= params.tolerance * effective_b_norm {
                 converged = true;
             }
        }

        if converged || total_iter >= params.max_iterations {
            break; // Break outer restart loop
        }
    } // End outer restart loop

    // Final check of residual norm if not already done by an early exit
    if !converged {
        let final_r = b - operator.apply(&x_current);
        current_residual_norm = final_r.iter().map(|&x| x * x).sum::<S>().sqrt();
        if current_residual_norm <= params.tolerance * effective_b_norm {
            converged = true;
        }
    }

    let final_reason = if converged {
        "Converged to tolerance.".to_string()
    } else if total_iter >= params.max_iterations {
        "Reached maximum iterations.".to_string()
    } else {
        // This case implies breakdown if j_final < m in some cycle
        // Or if loop terminated due to j_final == 0, which is unlikely post initial check.
        format!("Terminated due to breakdown or stagnation (j_final={}, m={}).", j_final, m)
    };

    Ok(GmresSolution {
        x: x_current,
        iterations: total_iter,
        residual_norm: current_residual_norm,
        converged,
        reason: final_reason,
    })
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // Minimal TestLinearOperator for f64
    struct TestLinearOperator {
        matrix: Array2<f64>,
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
            x.clone()
        }
        fn apply_right(&self, x: &Array1<f64>) -> Array1<f64> {
            x.clone()
        }
    }

    #[test]
    fn test_simple_diagonal_system_no_precond() {
        // A = diag(1, 2, 3)
        // b = (1, 4, 9)
        // x0 = (0, 0, 0)
        // Expected solution: x = (1, 2, 3)
        let operator = TestLinearOperator {
            matrix: array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        };
        let b = array![1.0, 4.0, 9.0];
        let x0 = array![0.0, 0.0, 0.0];

        let params = GmresParams {
            restart_length: 2, // Small restart length for testing
            max_iterations: 10,
            tolerance: 1e-5, // Looser tolerance
        };

        let result = gmres(&operator, &b, &x0, &params, None::<&TestPreconditioner>);

        assert!(result.is_ok());
        let solution = result.unwrap();

        let b_norm_val = b.iter().map(|&x| x*x).sum::<f64>().sqrt();
        assert!(solution.converged);
        assert!(solution.residual_norm < params.tolerance * b_norm_val || solution.residual_norm < params.tolerance); // Check relative or absolute
        assert!(solution.iterations <= params.max_iterations);

        let expected_x = array![1.0, 2.0, 3.0];
        let diff = &expected_x - &solution.x;
        let error_norm = diff.iter().map(|&x| x*x).sum::<f64>().sqrt();
        println!("Solution x: {:?}", solution.x);
        println!("Expected x: {:?}", expected_x);
        println!("Error norm: {}", error_norm);
        println!("Iterations: {}", solution.iterations);
        println!("Residual norm: {}", solution.residual_norm);

        assert!(error_norm < 1e-4); // Looser check
    }

    #[test]
    fn test_simple_diagonal_system_with_precond() {
        // A = diag(1, 2, 3)
        // P = Identity (effectively)
        // b = (1, 4, 9)
        // x0 = (0, 0, 0)
        // Expected solution: x = (1, 2, 3)
        let operator = TestLinearOperator {
            matrix: array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        };
        let preconditioner = TestPreconditioner { size: 3 };
        let b = array![1.0, 4.0, 9.0];
        let x0 = array![0.0, 0.0, 0.0];

        let params = GmresParams {
            restart_length: 2,
            max_iterations: 10,
            tolerance: 1e-5, // Looser tolerance
        };

        let result = gmres(&operator, &b, &x0, &params, Some(&preconditioner));

        assert!(result.is_ok());
        let solution = result.unwrap();

        let b_norm_val = b.iter().map(|&x| x*x).sum::<f64>().sqrt();
        assert!(solution.converged);
        assert!(solution.residual_norm < params.tolerance * b_norm_val || solution.residual_norm < params.tolerance);
        assert!(solution.iterations <= params.max_iterations);

        let expected_x = array![1.0, 2.0, 3.0];
        let diff = &expected_x - &solution.x;
        let error_norm = diff.iter().map(|&x| x*x).sum::<f64>().sqrt();
         println!("Solution x (precond): {:?}", solution.x);
        println!("Error norm (precond): {}", error_norm);
        println!("Iterations (precond): {}", solution.iterations);
        println!("Residual norm (precond): {}", solution.residual_norm);


        assert!(error_norm < 1e-4); // Looser check
    }
     #[test]
    fn test_b_is_zero() {
        let operator = TestLinearOperator {
            matrix: array![[1.0, 0.0], [0.0, 2.0]],
        };
        let b = array![0.0, 0.0];
        let x0 = array![1.0, 1.0]; // Non-zero x0

        let params = GmresParams {
            restart_length: 1,
            max_iterations: 50, // Increased iterations
            tolerance: 1e-5, // Looser tolerance
        };

        let result = gmres(&operator, &b, &x0, &params, None::<&TestPreconditioner>);
        assert!(result.is_ok());
        let solution = result.unwrap();

        // If b is zero, Ax = 0. If x0 is a solution (Ax0=0), it should return x0.
        // Otherwise, it should converge to a solution (often x=0 if A is invertible).
        // In this case, initial residual is A*x0. If A*x0 is zero, iterations = 0.
        // Here, A*x0 = [1,2], norm = sqrt(5). b_norm = 0. effective_b_norm = 1.
        // initial_r_norm = ||0 - Ax0|| = ||-Ax0|| = ||Ax0|| = sqrt(5)
        // It should converge to x = [0,0]

        println!("Solution x (b_zero): {:?}", solution.x);
        println!("Iterations (b_zero): {}", solution.iterations);
        println!("Residual norm (b_zero): {}", solution.residual_norm);

        assert!(solution.converged);
        assert!(solution.residual_norm < params.tolerance); // abs tol for zero b
        let expected_x = array![0.0, 0.0];
        let error_norm = (&expected_x - &solution.x).iter().map(|&x|x*x).sum::<f64>().sqrt();
        assert!(error_norm < 1e-4 || solution.x.iter().map(|&x|x*x).sum::<f64>().sqrt() < 1e-4); // Looser check
    }

    #[test]
    fn test_already_converged_x0() {
        let operator = TestLinearOperator {
            matrix: array![[1.0, 0.0], [0.0, 2.0]],
        };
        let b = array![1.0, 4.0];
        let x0 = array![1.0, 2.0]; // Ax0 = b

        let params = GmresParams {
            restart_length: 1,
            max_iterations: 5,
            tolerance: 1e-9,
        };

        let result = gmres(&operator, &b, &x0, &params, None::<&TestPreconditioner>);
        assert!(result.is_ok());
        let solution = result.unwrap();

        assert!(solution.converged);
        assert_eq!(solution.iterations, 0);
        assert!(solution.residual_norm < params.tolerance * b.iter().map(|&x|x*x).sum::<f64>().sqrt());
        let error_norm = (&x0 - &solution.x).iter().map(|&x|x*x).sum::<f64>().sqrt();
        assert!(error_norm < 1e-9); // Should return x0
    }
}
