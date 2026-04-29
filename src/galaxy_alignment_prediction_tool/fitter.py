import sys
import numpy as np
import logging

class fitter:
    def __init__(
        self,
        logger=logging.getLogger(__name__),
    ):
        """
        Class that handles fitting routines for power spectra, correlation functions, etc.
        """
        self.logger = logger


    def project_model_to_data(
        self, 
        r_data: np.ndarray, 
        model: np.ndarray, 
        r_model: np.ndarray
    ) -> np.ndarray:
        """Simple interpolator to project the model to the data scales. Scaling needs to be manually added afterwards.
        Args:
            r_data (np.array): data scale array
            model (np.array): model prediction array
            r_model (np.array): model scale array

        Returns:
            np.array: model vector projected to data scales
        """
        return np.interp(r_data, r_model, model)

    def _scale_cut(
        self, 
        r_min: float, 
        r_max: float, 
        r: np.ndarray, 
        data: np.ndarray, 
        cov: np.ndarray, 
        return_mask: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = (r > r_min) & (r < r_max)
        r_cut = r[mask]
        data_cut = data[mask]
        cov_cut = cov[np.ix_(mask, mask)]
        if return_mask:
            return r_cut, data_cut, cov_cut, mask
        else:
            return r_cut, data_cut, cov_cut
    
    def _renormalise_input(
        self, 
        data: np.ndarray, 
        model: np.ndarray, 
        cov: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        # Renormalise covariance and data vector
        sigma = np.sqrt(np.diagonal(cov))
        data_renorm = data/sigma
        model_renorm = model/sigma
        covariance_renorm = cov/sigma[:,None]/sigma[None,:]
        
        return data_renorm, model_renorm, covariance_renorm

    def _chi2_from_SVD(
        self,
        n_jk: int,
        cov: np.ndarray,
        data: np.ndarray,
        model: np.ndarray,
    ) -> float:
        
        # SVD decomposition
        U, s2, VT = np.linalg.svd(cov)

        # Rotate data and model into new basis
        data_rebase = U.T @ data
        model_rebase = U.T @ model

        # Filter out small singular values
        filtering_low_svd_bool = np.sqrt(2/n_jk) < s2
        data_rebase = data_rebase[filtering_low_svd_bool]
        model_rebase = model_rebase[filtering_low_svd_bool]
        s2 = s2[filtering_low_svd_bool]

        # Calculate chi2 in new basis
        chi2 = np.sum((data_rebase - model_rebase)**2 / s2)  

        return chi2

    def return_npoints_after_SVD(
        self,
        n_jk: int,
        cov: np.ndarray,
    ) -> int:
        """
        Simple wrapper to return number of points after SVD filtering
        Args:
            n_jk (int): number of jackknife samples
            cov (np.ndarray): covariance matrix
        Returns:
            int: number of points remaining after SVD filtering
        """

        U, s2, VT = np.linalg.svd(cov)
        filtering_low_svd_bool = np.sqrt(2/n_jk) < s2
        
        return np.sum(filtering_low_svd_bool)

    def zeropad_covariance_matrix(
        self,
        top_left: np.ndarray, 
        bottom_right: np.ndarray,
    ) -> np.ndarray:
        zero_crosscovar = np.zeros((top_left.shape[0], bottom_right.shape[1]))
        full_covariance = np.block([
            [top_left, zero_crosscovar],
            [zero_crosscovar.T, bottom_right],
        ])
        return full_covariance

    def get_chi2_full_covariance(
        self, 
        A_IA: float,
        b_g: float,
        r_data: list[np.ndarray],
        data: list[np.ndarray],
        r_model: list[np.ndarray],
        model: list[np.ndarray],
        cov: np.ndarray,
        fitting_range: list = [5, 16],
        projection_type_list: list[str] = ['gg', 'gp'],
        renormalise_input: bool = False,
        chi2_from_svd: bool = False,
        n_jk=None,
    ) -> float:
        """Calculate chi-squared from full covariance matrix for multiple projection types.
        
        Combines multiple data and model vectors (e.g., galaxy-galaxy and galaxy-matter power spectra)
        into a single vector, applies scaling factors, and computes chi-squared accounting for the 
        full covariance structure.

        Args:
            A_IA (float): Intrinsic alignment amplitude parameter.
            b_g (float): Galaxy bias parameter.
            r_data (list[np.ndarray]): List of scale arrays for data, one per projection type.
            data (list[np.ndarray]): List of data vectors, one per projection type.
            r_model (list[np.ndarray]): List of scale arrays for models, one per projection type.
            model (list[np.ndarray]): List of model prediction vectors, one per projection type.
            cov (np.ndarray): Full covariance matrix for concatenated data vector.
            fitting_range (list, optional): Scale range [r_min, r_max] for fitting. Defaults to [5, 16].
            projection_type_list (list[str], optional): Projection types ('gg', 'gp', etc.). Defaults to ['gg', 'gp'].
            renormalise_input (bool, optional): Whether to renormalize data and model by diagonal errors. Defaults to False.
            chi2_from_svd (bool, optional): Whether to use SVD-based chi-squared calculation. Defaults to False.
            n_jk (int, optional): Number of jackknife samples (required if chi2_from_svd=True). Defaults to None.

        Returns:
            float: Chi-squared value.
        """

        # Loop over projection types and build full data vector and covariance matrix
        model_list = []
        for idx, projection_type in enumerate(projection_type_list):
            # Project model to data scales
            model_like_data = self.project_model_to_data(r_data[idx], model[idx], r_model[idx])

            # Scale model with A_IA and b_g
            if projection_type == 'gp':
                model_like_data *= A_IA*b_g
            elif projection_type == 'gg':
                model_like_data *= b_g**2

            model_list.append(model_like_data)

        # Combine data and model into single vector
        model_projected = np.concatenate(model_list)
        data_concat = np.concatenate(data)
        r_data_concat = np.concatenate(r_data)

        if renormalise_input:
            data_renorm, model_renorm, cov_renorm = self._renormalise_input(data_concat, model_projected, cov)
        else:
            data_renorm, model_renorm, cov_renorm = data_concat, model_projected, cov

        # Cut data to relevant scales
        _, data_cut, cov_cut, mask = self._scale_cut(fitting_range[0], fitting_range[1], r_data_concat, data_renorm, cov_renorm, return_mask=True)
        model_cut = model_renorm[mask]

        # Calculate chi2
        if chi2_from_svd:
            chi2 = self._chi2_from_SVD(
                n_jk=n_jk,
                cov=cov_cut,
                data=data_cut,
                model=model_cut,
            )
        else:
            chi2 = (data_cut - model_cut) @ np.linalg.pinv(cov_cut) @ (data_cut - model_cut)
        
        return chi2

    def return_n_fitpoints(
        self, 
        r_data: list[np.ndarray],
        cov: np.ndarray,
        fitting_range: list = [5, 16],
        renormalise_input: bool = False,
        chi2_from_svd: bool = False,
        n_jk=None,
    ) -> int:
        
        """Wrapper to print out number of measurement points given a the analysis settings. Relevant for d.o.f. calculations.
        Does essentially the same as the get_chi2_full_covariance() method, except for the chi2 calculation.

        Args:
            r_data (list[np.ndarray]): List of scale arrays for data, one per projection type.
            cov (np.ndarray): Full covariance matrix for concatenated data vector.
            fitting_range (list, optional): Scale range [r_min, r_max] for fitting. Defaults to [5, 16].
            renormalise_input (bool, optional): Whether to renormalize data and model by diagonal errors. Defaults to False.
            chi2_from_svd (bool, optional): Whether to use SVD-based chi-squared calculation. Defaults to False.
            n_jk (int, optional): Number of jackknife samples (required if chi2_from_svd=True). Defaults to None.

        Returns:
            int: Number of data points that will be used in the fit

        """
        # Combine data and model into single vector
        r_data_concat = np.concatenate(r_data)
        data_stub = np.ones(len(r_data_concat))
        model_stub = np.ones(len(r_data_concat))

        if renormalise_input:
            data_renorm, _, cov_renorm = self._renormalise_input(data_stub, model_stub, cov)
        else:
            data_renorm, _, cov_renorm = data_stub, model_stub, cov
        
        _, _, cov_cut = self._scale_cut(fitting_range[0], fitting_range[1], r_data_concat, data_renorm, cov_renorm, return_mask=False)

        if chi2_from_svd:
            n_fitpoints = self.return_npoints_after_SVD(
                n_jk=n_jk,
                cov=cov_cut,
            )
        else:
            n_fitpoints = len(cov_cut)

        return n_fitpoints
        
    def get_chi2_full_covariance_with_NL_scaling(
        self, 
        A_IA: float,
        b_g: float,
        alpha_NLgg: float,
        alpha_NLgp: float,
        r_data: list[np.ndarray],
        data: list[np.ndarray],
        r_model: list[np.ndarray],
        model_linear: list[np.ndarray],
        model_nonlinear: list[np.ndarray],
        cov: np.ndarray,
        fitting_range: list = [[5, 16], [5, 16]],
        projection_type_list: list[str] = ['gg', 'gp'],
        renormalise_input: bool = False,
        chi2_from_svd: bool = False,
        n_jk=None,
    ) -> float:
        """Calculate chi-squared from full covariance matrix for multiple projection types.
        
        Combines multiple data and model vectors (e.g., galaxy-galaxy and galaxy-matter power spectra)
        into a single vector, applies scaling factors, and computes chi-squared accounting for the 
        full covariance structure.

        Args:
            A_IA (float): Intrinsic alignment amplitude parameter.
            b_g (float): Galaxy bias parameter.
            alpha_NLgg (float): Non-linear scaling parameter for galaxy-galaxy correlation.
            alpha_NLgp (float): Non-linear scaling parameter for galaxy-matter correlation.
            r_data (list[np.ndarray]): List of scale arrays for data, one per projection type.
            data (list[np.ndarray]): List of data vectors, one per projection type.
            r_model (list[np.ndarray]): List of scale arrays for models, one per projection type.
            model_linear (list[np.ndarray]): List of linear model prediction vectors, one per projection type.
            model_nonlinear (list[np.ndarray]): List of nonlinear model prediction vectors, one per projection type.
            cov (np.ndarray): Full covariance matrix for concatenated data vector.
            fitting_range (list, optional): Scale range [r_min, r_max] for fitting. Defaults to [[5, 16], [5, 16]].
            projection_type_list (list[str], optional): Projection types ('gg', 'gp', etc.). Defaults to ['gg', 'gp'].
            renormalise_input (bool, optional): Whether to renormalize data and model by diagonal errors. Defaults to False.
            chi2_from_svd (bool, optional): Whether to use SVD-based chi-squared calculation. Defaults to False.
            n_jk (int, optional): Number of jackknife samples (required if chi2_from_svd=True). Defaults to None.

        Returns:
            float: Chi-squared value.
        """

        # Loop over projection types and build full data vector and covariance matrix
        model_list = []
        for idx, projection_type in enumerate(projection_type_list):
            # Project model to data scales
            model_linear_like_data = self.project_model_to_data(r_data[idx], model_linear[idx], r_model[idx])
            model_nonlinear_like_data = self.project_model_to_data(r_data[idx], model_nonlinear[idx], r_model[idx])

            # Scale model with A_IA and b_g
            if projection_type == 'gp':
                model_like_data = model_linear_like_data + alpha_NLgp*(model_nonlinear_like_data - model_linear_like_data)
                model_like_data *= A_IA*b_g
            elif projection_type == 'gg':
                model_like_data = model_linear_like_data + alpha_NLgg*(model_nonlinear_like_data - model_linear_like_data)
                model_like_data *= b_g**2

            model_list.append(model_like_data)

        # Combine data and model into single vector
        model_projected = np.concatenate(model_list)
        data_concat = np.concatenate(data)
        r_data_concat = np.concatenate(r_data)

        if renormalise_input:
            data_renorm, model_renorm, cov_renorm = self._renormalise_input(data_concat, model_projected, cov)
        else:
            data_renorm, model_renorm, cov_renorm = data_concat, model_projected, cov

        # Cut data to relevant scales
        _, data_cut, cov_cut, mask = self._scale_cut(fitting_range[0], fitting_range[1], r_data_concat, data_renorm, cov_renorm, return_mask=True)
        model_cut = model_renorm[mask]

        # Calculate chi2
        if chi2_from_svd:
            chi2 = self._chi2_from_SVD(
                n_jk=n_jk,
                cov=cov_cut,
                data=data_cut,
                model=model_cut,
            )
        else:
            chi2 = (data_cut - model_cut) @ np.linalg.pinv(cov_cut) @ (data_cut - model_cut)
        
        return chi2

    def fit_Aia_bg_to_data(
            self, 
            r_data, 
            data,
            r_model,
            model,
            cov,
            fitting_range: list = [5, 16], 
            use_SVD: bool = False,
            n_jk: int = None,
            return_chi2: bool = False,
            include_Hartlap_correction: bool = False,
        ):

        # Cut data to relevant scales and then project model to data
        r_data, data, cov = self._scale_cut(fitting_range[0], fitting_range[1], r_data, data, cov)
        model = self.project_model_to_data(r_data, model, r_model)

        # Renormalise input for more robust fitting
        data_renorm, model_renorm, covariance_renorm = self._renormalise_input(data, model, cov)

        if use_SVD:
            if n_jk is None:
                raise ValueError("If using SVD, you must specify the number of jackknife samples.")

            # SVD decomposition
            u, s2, vT = np.linalg.svd(covariance_renorm)

            # Rotate data and model into new basis
            data_rebase = u.T @ data_renorm
            model_rebase = u.T @ model_renorm

            # Filter out small singular values
            filtering_low_svd_bool = np.sqrt(2/n_jk) < s2
            data_rebase = data_rebase[filtering_low_svd_bool]
            model_rebase = model_rebase[filtering_low_svd_bool]
            s2 = s2[filtering_low_svd_bool]

            self.logger.info(f"Points used for fitting: {len(data_rebase)}")

            # Defining fitting function
            data_func = lambda x, scaling: scaling*model_rebase

            # Scale covariance with Hartlap correction if requested
            if include_Hartlap_correction:
                hartlap_factor = (n_jk - len(data_rebase) - 2)/(n_jk -1)
                self.logger.info(f"Hartlap factor applied to covariance: {hartlap_factor}")
                s2 /= hartlap_factor


            fit_params, fit_cov = curve_fit(data_func, np.ones_like(data_rebase), data_rebase, p0=[0.5], sigma=np.sqrt(s2), absolute_sigma=True)

            # Calculate chi2
            chi2 = (data_rebase - model_rebase*fit_params[0]) @ np.linalg.inv(np.diag(s2)) @ (data_rebase - model_rebase*fit_params[0])
            reduced_chi2 = chi2/(len(data_rebase) - len(fit_params))

        else:
            # Scale covariance with Hartlap correction if requested
            if include_Hartlap_correction:
                hartlap_factor = (n_jk - len(data_renorm) - 2)/(n_jk -1)
                self.logger.info(f"Hartlap factor applied to covariance: {hartlap_factor}")
                covariance_renorm /= hartlap_factor

            # Fit with scipy
            data_func = lambda x, scaling: scaling*model_renorm
            fit_params, fit_cov = curve_fit(data_func, np.ones_like(data_renorm), data_renorm, p0=[1], sigma=covariance_renorm, absolute_sigma=True)

            # Calculate chi2
            chi2 = (data_renorm - model_renorm*fit_params[0]) @ np.linalg.inv(covariance_renorm) @ (data_renorm - model_renorm*fit_params[0])
            reduced_chi2 = chi2/(len(data_renorm) - len(fit_params))

        self.logger.info(f"Fitted scaling: A_IA*b_g = {fit_params[0]}+-{np.sqrt(fit_cov[0][0])}")
        self.logger.info(f"Reduced chi2: {reduced_chi2}")

        if return_chi2:
            return fit_params[0], np.sqrt(fit_cov[0][0]), reduced_chi2
        else:
            return fit_params[0], np.sqrt(fit_cov[0][0])