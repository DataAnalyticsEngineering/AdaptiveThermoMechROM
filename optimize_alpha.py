import numpy as np
import numpy.linalg as la

def naive(alpha, sampling_C, sampling_eps, ref_C, ref_eps):
    opt_C = np.empty_like(ref_C)
    opt_eps = np.empty_like(ref_eps)
    interpolate = lambda alpha, x1, x2: x1 + alpha * (x2 - x1)

    for idx, (C_samples, eps_samples, C_ref, eps_ref) in enumerate(zip(sampling_C, sampling_eps, ref_C, ref_eps)):
        opt_C[idx] = interpolate(alpha, *C_samples)
        opt_eps[idx] = interpolate(alpha, *eps_samples)

    return opt_C, opt_eps

def opt1(sampling_C, sampling_eps, ref_C, ref_eps):
    opt_C = np.empty_like(ref_C)
    opt_eps = np.empty_like(ref_eps)
    interpolate = lambda alpha, x1, x2: x1 + alpha * (x2 - x1)
    explicit_minimizer_numerator = lambda Cr, C1, C2: ((Cr - C1).flatten() @ (C2 - C1).flatten())
    explicit_minimizer_denominator = lambda Cr, C1, C2: ((C2 - C1).flatten() @ (C2 - C1).flatten())

    stiffness_weight = 0.1
    numerator, denominator = 0, 0

    for C_samples, eps_samples, C_ref, eps_ref in zip(sampling_C, sampling_eps, ref_C, ref_eps):
        C_normalization = C_ref.flatten() @ C_ref.flatten()
        eps_normalization = eps_ref.flatten() @ eps_ref.flatten()

        numerator += stiffness_weight * (explicit_minimizer_numerator(C_ref, *C_samples) / C_normalization)
        denominator += stiffness_weight * (explicit_minimizer_denominator(C_ref, *C_samples) / C_normalization)
        numerator += (1 - stiffness_weight) * (explicit_minimizer_numerator(eps_ref, *eps_samples) / eps_normalization)
        denominator += (1 - stiffness_weight) * (explicit_minimizer_denominator(eps_ref, *eps_samples) / eps_normalization)

    opt_alpha = numerator / denominator

    for idx, (C_samples, eps_samples, C_ref, eps_ref) in enumerate(zip(sampling_C, sampling_eps, ref_C, ref_eps)):
        opt_C[idx] = interpolate(opt_alpha, *C_samples)
        opt_eps[idx] = interpolate(opt_alpha, *eps_samples)

    return opt_C, opt_eps

def opt2(sampling_C, sampling_eps, ref_C, ref_eps):
    opt_C = np.empty_like(ref_C)
    opt_eps = np.empty_like(ref_eps)
    interpolate = lambda alpha, x1, x2: x1 + alpha * (x2 - x1)
    explicit_minimizer_numerator = lambda Cr, C1, C2: ((Cr - C1).flatten() @ (C2 - C1).flatten())
    explicit_minimizer_denominator = lambda Cr, C1, C2: ((C2 - C1).flatten() @ (C2 - C1).flatten())

    numerator_c, denominator_c, numerator_eps, denominator_eps = 0, 0, 0, 0

    for C_samples, eps_samples, C_ref, eps_ref in zip(sampling_C, sampling_eps, ref_C, ref_eps):
        C_normalization = C_ref.flatten() @ C_ref.flatten()
        eps_normalization = eps_ref.flatten() @ eps_ref.flatten()

        numerator_c += explicit_minimizer_numerator(C_ref, *C_samples) / C_normalization
        denominator_c += explicit_minimizer_denominator(C_ref, *C_samples) / C_normalization
        numerator_eps += explicit_minimizer_numerator(eps_ref, *eps_samples) / eps_normalization
        denominator_eps += explicit_minimizer_denominator(eps_ref, *eps_samples) / eps_normalization

    opt_alpha_C = numerator_c / denominator_c
    opt_alpha_eps = numerator_eps / denominator_eps

    for idx, (C_samples, eps_samples, C_ref, eps_ref) in enumerate(zip(sampling_C, sampling_eps, ref_C, ref_eps)):
        opt_C[idx] = interpolate(opt_alpha_C, *C_samples)
        opt_eps[idx] = interpolate(opt_alpha_eps, *eps_samples)

    return opt_C, opt_eps

def opt4(sampling_C, sampling_eps, ref_C, ref_eps):
    opt_C = np.empty_like(ref_C)
    opt_eps = np.empty_like(ref_eps)
    interpolate = lambda alpha, x1, x2: x1 + alpha * (x2 - x1)
    explicit_minimizer = lambda Cr, C1, C2: ((Cr - C1).flatten() @ (C2 - C1).flatten()) / (
        (C2 - C1).flatten() @ (C2 - C1).flatten())

    for idx, (C_samples, eps_samples, C_ref, eps_ref) in enumerate(zip(sampling_C, sampling_eps, ref_C, ref_eps)):
        opt_alpha_C = explicit_minimizer(C_ref, *C_samples)
        opt_alpha_eps = explicit_minimizer(eps_ref, *eps_samples)
        opt_C[idx] = interpolate(opt_alpha_C, *C_samples)
        opt_eps[idx] = interpolate(opt_alpha_eps, *eps_samples)
    return opt_C, opt_eps

def opt4_alphas(sampling_C, sampling_eps, ref_C, ref_eps):
    opt_alpha_C = np.empty(ref_C.shape[0])
    opt_alpha_eps = np.empty(ref_eps.shape[0])
    explicit_minimizer = lambda Cr, C1, C2: ((Cr - C1).flatten() @ (C2 - C1).flatten()) / (
        (C2 - C1).flatten() @ (C2 - C1).flatten())

    for idx, (C_samples, eps_samples, C_ref, eps_ref) in enumerate(zip(sampling_C, sampling_eps, ref_C, ref_eps)):
        opt_alpha_C[idx] = explicit_minimizer(C_ref, *C_samples)
        opt_alpha_eps[idx] = explicit_minimizer(eps_ref, *eps_samples)

    return opt_alpha_C[:, None, None], opt_alpha_eps[:, None, None]
