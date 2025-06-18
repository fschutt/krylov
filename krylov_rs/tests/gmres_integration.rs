// Allow dead code for now as not all common utils might be used immediately
#![allow(dead_code)]

mod common;
use common::*;

use krylov_rs::gmres::{gmres, GmresParams};
use krylov_rs::operator::LinearOperator; // For calling apply on operator if needed for x_exact
use ndarray::{array, Array1}; // Removed Ix1
// num_traits individual imports not needed if TestScalar is f64 and methods are inherent
// use num_traits::{Float, FromPrimitive, Zero}; // For f64 specifically

type TestScalar = f64; // Using f64 for tests

#[test]
fn test_gmres_spd_matrix_no_precond() {
    let size = 3;
    let op = create_spd_matrix::<TestScalar>(size);
    let x_exact = array![1.0, 2.0, 3.0];
    let b = op.apply(&x_exact); // b = A * x_exact

    let x0 = Array1::zeros(size);
    let params = GmresParams {
        restart_length: size, // Should converge fast for SPD with full restart
        max_iterations: 2 * size, // Generous max_iterations
        tolerance: 1e-9,
    };

    let result = gmres(&op, &b, &x0, &params, None::<&IdentityPreconditioner<TestScalar>>);

    assert!(result.is_ok(), "GMRES failed: {:?}", result.err());
    let solution = result.unwrap();

    assert!(solution.converged, "GMRES did not converge. Reason: {}", solution.reason);
    let error_norm = l2_norm_manual(&(&solution.x - &x_exact));
    println!("GMRES SPD NoPrecond: x = {:?}, expected_x = {:?}, error_norm = {}", solution.x, x_exact, error_norm);
    println!("Iterations: {}, Residual Norm: {}", solution.iterations, solution.residual_norm);
    assert!(error_norm < 1e-7, "Solution accuracy too low.");
    assert!(solution.residual_norm < params.tolerance * l2_norm_manual(&b) || solution.residual_norm < params.tolerance, "Residual norm condition not met.");
}

#[test]
fn test_gmres_spd_matrix_with_precond() {
    let size = 3;
    let op = create_spd_matrix::<TestScalar>(size);
    let x_exact = array![1.0, 2.0, 3.0];
    let b = op.apply(&x_exact);
    let preconditioner = IdentityPreconditioner::<TestScalar>::new(size);

    let x0 = Array1::zeros(size);
    let params = GmresParams {
        restart_length: size,
        max_iterations: 2 * size,
        tolerance: 1e-9,
    };

    let result = gmres(&op, &b, &x0, &params, Some(&preconditioner));

    assert!(result.is_ok(), "GMRES failed: {:?}", result.err());
    let solution = result.unwrap();

    assert!(solution.converged, "GMRES did not converge with preconditioner. Reason: {}", solution.reason);
    let error_norm = l2_norm_manual(&(&solution.x - &x_exact));
    println!("GMRES SPD WithPrecond: x = {:?}, expected_x = {:?}, error_norm = {}", solution.x, x_exact, error_norm);
    println!("Iterations: {}, Residual Norm: {}", solution.iterations, solution.residual_norm);
    assert!(error_norm < 1e-7, "Solution accuracy too low with preconditioner.");
    assert!(solution.residual_norm < params.tolerance * l2_norm_manual(&b) || solution.residual_norm < params.tolerance, "Residual norm condition not met with preconditioner.");
}

#[test]
fn test_gmres_nonsymmetric_matrix_no_precond() {
    let size = 3;
    // Using a known non-symmetric matrix where solution might be simple
    // A = [[1, 2, 0], [0, 1, 3], [4, 0, 1]]
    let op_mat = array![[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [4.0, 0.0, 1.0]];
    let op = TestMatrix {matrix: op_mat };
    let x_exact = array![1.0, 1.0, 1.0]; // Simple exact solution
    let b = op.apply(&x_exact); // b = A * (1,1,1) = (3,4,5)

    let x0 = Array1::zeros(size);
    let params = GmresParams {
        restart_length: size,
        max_iterations: 2 * size,
        tolerance: 1e-9,
    };

    let result = gmres(&op, &b, &x0, &params, None::<&IdentityPreconditioner<TestScalar>>);

    assert!(result.is_ok(), "GMRES failed for non-symmetric: {:?}", result.err());
    let solution = result.unwrap();

    assert!(solution.converged, "GMRES did not converge for non-symmetric matrix. Reason: {}", solution.reason);
    let error_norm = l2_norm_manual(&(&solution.x - &x_exact));
    println!("GMRES NonSymm: x = {:?}, expected_x = {:?}, error_norm = {}", solution.x, x_exact, error_norm);
    println!("Iterations: {}, Residual Norm: {}", solution.iterations, solution.residual_norm);
    assert!(error_norm < 1e-7, "Solution accuracy too low for non-symmetric.");
     assert!(solution.residual_norm < params.tolerance * l2_norm_manual(&b) || solution.residual_norm < params.tolerance, "Residual norm condition not met for non-symmetric.");
}

// Note: CirculantOperator is now in common.rs
#[test]
fn test_gmres_matrix_free_circulant() {
    let size = 3;
    // c = [2, -1, 0] -> A = [[2, 0, -1], [-1, 2, 0], [0, -1, 2]] (This is SPD, Toeplitz, and Circulant)
    let circulant_vec = array![2.0, -1.0, 0.0];
    let op = CirculantOperator::new(circulant_vec);

    let x_exact = array![1.0, 2.0, 3.0];
    let b = op.apply(&x_exact);

    let x0 = Array1::zeros(size);
    let params = GmresParams {
        restart_length: size,
        max_iterations: 2 * size,
        tolerance: 1e-9,
    };

    let result = gmres(&op, &b, &x0, &params, None::<&IdentityPreconditioner<TestScalar>>);

    assert!(result.is_ok(), "GMRES failed for matrix-free: {:?}", result.err());
    let solution = result.unwrap();

    assert!(solution.converged, "GMRES did not converge for matrix-free operator. Reason: {}", solution.reason);
    let error_norm = l2_norm_manual(&(&solution.x - &x_exact));
    println!("GMRES MatrixFree: x = {:?}, expected_x = {:?}, error_norm = {}", solution.x, x_exact, error_norm);
    println!("Iterations: {}, Residual Norm: {}", solution.iterations, solution.residual_norm);
    assert!(error_norm < 1e-7, "Solution accuracy too low for matrix-free.");
    assert!(solution.residual_norm < params.tolerance * l2_norm_manual(&b) || solution.residual_norm < params.tolerance, "Residual norm condition not met for matrix-free.");
}
