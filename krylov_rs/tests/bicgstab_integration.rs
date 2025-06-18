// Allow dead code for now as not all common utils might be used immediately
#![allow(dead_code)]

mod common;
use common::*;

use krylov_rs::bicgstab::{bicgstab, BicgstabParams};
use krylov_rs::operator::LinearOperator; // For op.apply for b
use ndarray::{array, Array1};
// num_traits individual imports not needed if TestScalar is f64
// use num_traits::{Float, FromPrimitive, Zero};

type TestScalar = f64;

#[test]
fn test_bicgstab_spd_matrix_no_precond() {
    let size = 3;
    let op = create_spd_matrix::<TestScalar>(size);
    let x_exact = array![1.0, 2.0, 3.0];
    let b = op.apply(&x_exact);

    let x0 = Array1::zeros(size);
    let params = BicgstabParams {
        max_iterations: 2 * size, // BiCGSTAB often converges faster than GMRES for SPD
        tolerance: 1e-9,
    };

    let result = bicgstab(&op, &b, &x0, &params, None::<&IdentityPreconditioner<TestScalar>>);

    assert!(result.is_ok(), "BiCGSTAB failed: {:?}", result.err());
    let solution = result.unwrap();

    assert!(solution.converged, "BiCGSTAB did not converge. Reason: {}", solution.reason);
    let error_norm = l2_norm_manual(&(&solution.x - &x_exact));
    println!("BiCGSTAB SPD NoPrecond: x = {:?}, expected_x = {:?}, error_norm = {}", solution.x, x_exact, error_norm);
    println!("Iterations: {}, Residual Norm: {}", solution.iterations, solution.residual_norm);
    assert!(error_norm < 1e-7, "Solution accuracy too low.");
    assert!(solution.residual_norm < params.tolerance * l2_norm_manual(&b) || solution.residual_norm < params.tolerance, "Residual norm condition not met.");
}

#[test]
fn test_bicgstab_spd_matrix_with_precond() {
    let size = 3;
    let op = create_spd_matrix::<TestScalar>(size);
    let x_exact = array![1.0, 2.0, 3.0];
    let b = op.apply(&x_exact);
    let preconditioner = IdentityPreconditioner::<TestScalar>::new(size);

    let x0 = Array1::zeros(size);
    let params = BicgstabParams {
        max_iterations: 2 * size,
        tolerance: 1e-9,
    };

    let result = bicgstab(&op, &b, &x0, &params, Some(&preconditioner));

    assert!(result.is_ok(), "BiCGSTAB failed: {:?}", result.err());
    let solution = result.unwrap();

    assert!(solution.converged, "BiCGSTAB did not converge with preconditioner. Reason: {}", solution.reason);
    let error_norm = l2_norm_manual(&(&solution.x - &x_exact));
    println!("BiCGSTAB SPD WithPrecond: x = {:?}, expected_x = {:?}, error_norm = {}", solution.x, x_exact, error_norm);
    println!("Iterations: {}, Residual Norm: {}", solution.iterations, solution.residual_norm);
    assert!(error_norm < 1e-7, "Solution accuracy too low with preconditioner.");
    assert!(solution.residual_norm < params.tolerance * l2_norm_manual(&b) || solution.residual_norm < params.tolerance, "Residual norm condition not met with preconditioner.");
}

#[test]
fn test_bicgstab_nonsymmetric_matrix_no_precond() {
    let size = 3;
    // Using the same non-symmetric matrix as in GMRES tests
    // A = [[1, 2, 0], [0, 1, 3], [4, 0, 1]]
    let op_mat = array![[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [4.0, 0.0, 1.0]];
    let op = TestMatrix {matrix: op_mat };
    let x_exact = array![1.0, 1.0, 1.0]; // b = (3,4,5)
    let b = op.apply(&x_exact);

    let x0 = Array1::zeros(size);
    let params = BicgstabParams {
        max_iterations: 2 * size,
        tolerance: 1e-9,
    };

    let result = bicgstab(&op, &b, &x0, &params, None::<&IdentityPreconditioner<TestScalar>>);

    assert!(result.is_ok(), "BiCGSTAB failed for non-symmetric: {:?}", result.err());
    let solution = result.unwrap();

    assert!(solution.converged, "BiCGSTAB did not converge for non-symmetric matrix. Reason: {}", solution.reason);
    let error_norm = l2_norm_manual(&(&solution.x - &x_exact));
    println!("BiCGSTAB NonSymm: x = {:?}, expected_x = {:?}, error_norm = {}", solution.x, x_exact, error_norm);
    println!("Iterations: {}, Residual Norm: {}", solution.iterations, solution.residual_norm);
    assert!(error_norm < 1e-7, "Solution accuracy too low for non-symmetric.");
    assert!(solution.residual_norm < params.tolerance * l2_norm_manual(&b) || solution.residual_norm < params.tolerance, "Residual norm condition not met for non-symmetric.");
}

#[test]
fn test_bicgstab_matrix_free_circulant() {
    let size = 3;
    // Using the same circulant operator as in GMRES tests
    // c = [2, -1, 0] -> A = [[2, 0, -1], [-1, 2, 0], [0, -1, 2]]
    let circulant_vec = array![2.0, -1.0, 0.0];
    // Need to re-define CirculantOperator or ensure common.rs provides it to both test files.
    // For now, let's assume common.rs CirculantOperator is available.
    // If not, copy the definition here or (better) ensure common module structure allows sharing.
    // For this test, it's okay if CirculantOperator is re-defined if common.rs is not fully set up for cross-test-file use.
    // However, the instruction implies common.rs should be usable.
    let op = CirculantOperator::new(circulant_vec.clone()); // Use .clone() if circulant_vec is used elsewhere

    let x_exact = array![1.0, 2.0, 3.0];
    let b = op.apply(&x_exact);

    let x0 = Array1::zeros(size);
    let params = BicgstabParams {
        max_iterations: 2 * size,
        tolerance: 1e-9,
    };

    let result = bicgstab(&op, &b, &x0, &params, None::<&IdentityPreconditioner<TestScalar>>);

    assert!(result.is_ok(), "BiCGSTAB failed for matrix-free: {:?}", result.err());
    let solution = result.unwrap();

    assert!(solution.converged, "BiCGSTAB did not converge for matrix-free operator. Reason: {}", solution.reason);
    let error_norm = l2_norm_manual(&(&solution.x - &x_exact));
    println!("BiCGSTAB MatrixFree: x = {:?}, expected_x = {:?}, error_norm = {}", solution.x, x_exact, error_norm);
    println!("Iterations: {}, Residual Norm: {}", solution.iterations, solution.residual_norm);
    assert!(error_norm < 1e-7, "Solution accuracy too low for matrix-free.");
    assert!(solution.residual_norm < params.tolerance * l2_norm_manual(&b) || solution.residual_norm < params.tolerance, "Residual norm condition not met for matrix-free.");
}
