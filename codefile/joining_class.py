# -*- coding: utf-8 -*-
"""
Author:
    lprtk

Description:
    The objective is to create a tool that can combine predictions from two
    different models, whether it is a regression or classification task.
    Generally, when doing time series for example, we can use traditional
    econometric models or more sophisticated Machine and Deep Learning models.
    Generally, econometric models provide good short-term predictions but poor
    long-term predictions while some Deep Learning models provide very good
    long-term predictions. Thanks to these classes, it is now possible to merge
    two forecast vectors according to an exponential coefficient: we give more
    weight in the short term to the forecasts of the first model (the econometric
    model for example) and conversely, we give more weight to the forecasts of
    the second model in the long term (Deep Learning model for example).

License:
    MIT License
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#------------------------------------------------------------------------------


class JoiningRegressor:
    def __init__(self, y_pred_1, y_pred_2,
                 exponential_coeff: float=0.8) -> None:
        """
        Function that allows to build the JoiningRegressor class and initialise
        the parameters.

        Parameters
        ----------
        y_pred_1 : numpy.ndarray (1,) or pandas.cores.series.Series
            1d array corresponding to the first predictions to be joined. It is
            the predictions made by a short term model.

        y_pred_2 : numpy.ndarray (1,) or pandas.cores.series.Series
            1d array corresponding to the second predictions to be joined. It is
            the predictions made by a long term model.

        exponential_coeff : float, optional, default=0.8
            Exponential coefficient needed to model the exponential smoothing
            curve. A high coefficient will result in a steep slope and a high
            weight of the long-range forecasts. If the coefficient is low, the
            weight of the short-term forecasts will be more important than the
            long-term forecast because the slope of the exponential curve will
            be lower. Default is 0.8.

        Raises
        ------
        TypeError
            - y_pred_1 parameter must be an numpy.ndarray or pandas.cores.series.Series
            to use the functions associated with the JoiningRegressor class.

            - y_pred_2 parameter must be an numpy.ndarray or pandas.cores.series.Series
            to use the functions associated with the JoiningRegressor class.

            - exponential_coeff parameter must be a float to use the functions
            associated with the JoiningRegressor class.
            
        AssertionError
            - y_pred_1 and y_pred_2 must have the same shape to use the functions
            associated with the JoiningRegressor class.

        Returns
        -------
        None
            NoneType.

        """
        if isinstance(y_pred_1, np.ndarray):
            self.y_pred_1 = y_pred_1
        elif isinstance(y_pred_1, pd.core.series.Series):
            self.y_pred_1 = y_pred_1.to_numpy()    
        else:
            raise TypeError(
                f"'y_pred_1' parameter must be an ndarray: got {type(y_pred_1)}"
                )
        
        if isinstance(y_pred_2, np.ndarray):
            self.y_pred_2 = y_pred_2
        elif isinstance(y_pred_2, pd.core.series.Series):
            self.y_pred_2 = y_pred_2.to_numpy()
        else:
            raise TypeError(
                f"'y_pred_2' parameter must be an ndarray: got {type(y_pred_2)}"
                )
        
        assert self.y_pred_1.shape[0] == self.y_pred_2.shape[0],\
        "'y_pred_1' and 'y_pred_2' parameters must have the same shape"
        
        if isinstance(exponential_coeff, float):
            self.exponential_coeff = exponential_coeff
        else:
            raise TypeError(
                f"'exponential_coeff' parameter must be a float: got {type(exponential_coeff)}"
                )
        
        # calibration of the axes to compute the exponential curve and display the graphs
        self.x_axis = np.arange(0, self.y_pred_1.shape[0])
        self.y_axis = (np.power(self.exponential_coeff, self.x_axis))
        
        # creation of a null vector which will be filled with the new predictions
        self.y_pred = np.zeros(shape=self.y_pred_1.shape[0])
        
        # check is fitted validation
        self.fitted = False
    
    
    def fit(self) -> None:
        """
        Function that allows to fit the joinded model.

        Returns
        -------
        None
            NoneType.

        """
        # calculation of the new predictions weighted by the exponential coefficient
        for idx in range(0, self.y_pred_1.shape[0]):
            alpha = self.y_axis[idx]
            self.y_pred[idx] = ((alpha*self.y_pred_1[idx])+((1-alpha)*self.y_pred_2[idx]))
        
        self.fitted = True
    
    
    def get_predictions(self) -> np.ndarray:
        """
        Function that allows to return new predictions from the joined model.

        Raises
        ------
        ValueError
            Before using this function, the joined model must be fitted.

        Returns
        -------
        self.y_pred : numpy.ndarray
            New predictions from the joined model.

        """
        if self.fitted:
            return self.y_pred
        else:
            raise ValueError(
                "Estimator's instance is not fitted yet. Call '.fit()' before using this function"
            )
    
    
    def get_plot(self, date=None, figsize: tuple=(30, 8),
                 colors: list=["black", "red", "blue", "green"],
                 title: str="New predictions", xlabel: str="Frequency",
                 ylabel: str="Value", label1: str="1st predictions",
                 label2: str="2nd predictions", xticks_rotation: int=0) -> None:
        """
        Function that allows to display a subplots with the exponential smoothing
        curve and the different predictions (y_pred_1, y_pred_2 and the news).

        Parameters
        ----------
        date : None or pandas.core.series.Series, optional, default=None
            If date parameter is specified, it will be added as an index for
            the x-axis of the plots.

        figsize : tuple, optional, default=(30, 8)
            Size of the subplots in pixels. Default is (width=30px, height=8px).

        colors : list, optional, default=["black", "red", "blue", "green"]
            Color list for subplots. Default is ["black", "red", "blue", "green"].

        title : str, optional, default="New predictions"
            Subplot's title. Default is "New predictions".

        xlabel : str, optional, default="Frequency"
            X-axis title. Default is "Frequency".

        ylabel : str, optional, default="Value"
            Y-axis title. Default is "Value".
        
        label1 : str, optional, default="1st predictions"
            Title for y_pred_1 values. Default is "1st predictions".
            
        label2 : str, optional, default="2nd predictions"
            Title for y_pred_2 values. Default is "2nd predictions".

        xticks_rotation : int, optional, default=0
            Labels' orientation on the x-axis. Default is 0.

        Raises
        ------
        TypeError
            - date parameter must be None or a series to use get_plot function.
            
            - figsize parameter must be a tuple to use get_plot function.
            
            - colors parameter must be a list to use get_plot function.
            
            - title parameter must be a str to use get_plot function.
            
            - xlabel parameter must be a str to use get_plot function.
            
            - ylabel parameter must be a str to use get_plot function.
            
            - label1 parameter must be a str to use get_plot function.
            
            - label2 parameter must be a str to use get_plot function.
            
            - xticks_rotation parameter must be an int to use get_plot function.

        ValueError
            Before using this function, the joined model must be fitted.

        AssertionError
            - date parameter must have the same shape as y_pred_1 and y_pred_2
            to use get_plot function.
            
            - figsize parameter must contain only the height and width to use
            get_plot function.
            
            - colors parameter must contain only 4 colors to use get_plot function.

        Returns
        -------
        None
            NoneType.

        """
        if date is None or isinstance(date, pd.core.series.Series):
            assert self.y_pred_1.shape[0] == date.shape[0],\
            "'date' parameter must have the same shape as 'y_pred_1' and 'y_pred_2'"
        else:
            raise TypeError(
                f"'date' parameter must be None or a series: got {type(date)}"
            )
        
        if isinstance(figsize, tuple):
            assert len(figsize) == 2,\
            "'figsize' parameter must contain the height and width: figsize=(width, height)"
        else:
            raise TypeError(
                f"'figsize' parameter must be a tuple: got {type(figsize)}"
            )
        
        if isinstance(colors, list):
            assert len(colors) == 4, "'colors' parameter must contain 4 colors"
        else:
            raise TypeError(
                f"'colors' parameter must be a list: got {type(colors)}"
            )
        
        if isinstance(title, str):
            pass
        else:
            raise TypeError(
                f"'title' parameter must be a str: got {type(title)}"
            )
        
        if isinstance(xlabel, str):
            pass
        else:
            raise TypeError(
                f"'xlabel' parameter must be a str: got {type(xlabel)}"
            )
        
        if isinstance(ylabel, str):
            pass
        else:
            raise TypeError(
                f"'ylabel' parameter must a str: got {type(ylabel)}"
            )
        
        if isinstance(label1, str):
            pass
        else:
            raise TypeError(
                f"'label1' parameter must a str: got {type(label1)}"
            )
        
        if isinstance(label2, str):
            pass
        else:
            raise TypeError(
                f"'label2' parameter must be a str: got {type(label2)}"
            )
        
        if isinstance(xticks_rotation, int):
            pass
        else:
            raise TypeError(
                f"'xticks_rotation' parameter must be an int: got {type(xticks_rotation)}"
            )
        
        if self.fitted:
            if date is None:
                df_values = pd.concat(
                    [
                        pd.DataFrame(self.y_pred_1, columns=[label1]),
                        pd.DataFrame(self.y_pred_2, columns=[label2])
                    ],
                    axis=1
                )
                df_values = pd.concat(
                    [
                        df_values,
                        pd.DataFrame(self.y_pred, columns=["New predictions"])
                    ],
                    axis=1
                )
                df_values.reset_index(drop=True, inplace=True)
            else:
                self.x_axis = date.to_frame(name="Date")
                df_values = pd.concat(
                    [
                        date.to_frame(name="Date"),
                        pd.DataFrame(self.y_pred_1, columns=[label1])
                    ],
                    axis=1
                )
                df_values = pd.concat(
                    [
                        df_values,
                        pd.DataFrame(self.y_pred_2, columns=[label2])
                    ],
                    axis=1
                )
                df_values = pd.concat(
                    [
                        df_values,
                        pd.DataFrame(self.y_pred, columns=["New predictions"])
                    ],
                    axis=1
                )
                df_values.reset_index(drop=True, inplace=True)
                df_values.set_index(keys="Date", drop=True, inplace=True)
            
            fig = plt.figure(figsize=figsize)
            
            plt.subplot(1, 2, 1)
            plt.plot(self.x_axis, self.y_axis, color=colors[0], label="Exponential smoothing")
            plt.title(f"Exponential smoothing curve for {self.exponential_coeff} coefficient")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend(loc="best")
            plt.xticks(rotation=xticks_rotation)
            
            plt.subplot(1, 2, 2)
            df_values[label1].plot(color=colors[1], label=label1)
            df_values[label2].plot(color=colors[2], label=label2)
            df_values["New predictions"].plot(color=colors[3], label="New predictions")
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend(loc="best")
            plt.xticks(rotation=xticks_rotation)
            
            plt.subplots_adjust(hspace=0.3)
            plt.show()
        else:
            raise ValueError(
                "Estimator's instance is not fitted yet. Call '.fit()' before using this function"
            )


#------------------------------------------------------------------------------


class JoiningClassifier:
    def __init__(self, y_pred_1, y_pred_2,
                 exponential_coeff: float=0.8, threshold: float=0.5) -> None:
        """
        Function that allows to build the JoiningClassifier class and initialise
        the parameters.

        Parameters
        ----------
        y_pred_1 : numpy.ndarray (1,) or pandas.cores.series.Series
            1d array corresponding to the first predictions to be joined. It is
            the predictions made by a short term model.

        y_pred_2 : numpy.ndarray (1,) or pandas.cores.series.Series
            1d array corresponding to the second predictions to be joined. It is
            the predictions made by a long term model.

        exponential_coeff : float, optional, default=0.8
            Exponential coefficient needed to model the exponential smoothing
            curve. A high coefficient will result in a steep slope and a high
            weight of the long-range forecasts. If the coefficient is low, the
            weight of the short-term forecasts will be more important than the
            long-term forecast because the slope of the exponential curve will
            be lower. Default is 0.8.

        threshold : float, optional, default=0.5
            Classification cut-off. Default is 0.5.

        Raises
        ------
        TypeError
            - y_pred_1 parameter must be an numpy.ndarray or pandas.cores.series.Series
            to use the functions associated with the JoiningClassifier class.

            - y_pred_2 parameter must be an numpy.ndarray or pandas.cores.series.Series
            to use the functions associated with the JoiningClassifier class.

            - exponential_coeff parameter must be a float to use the functions
            associated with the JoiningClassifier class.

        AssertionError
            - y_pred_1 and y_pred_2 must have the same shape to use the functions
            associated with the JoiningClassifier class.

        Returns
        -------
        None
            NoneType.

        """
        if isinstance(y_pred_1, np.ndarray):
            self.y_pred_1 = y_pred_1
        elif isinstance(y_pred_1, pd.core.series.Series):
            self.y_pred_1 = y_pred_1.to_numpy()    
        else:
            raise TypeError(
                f"'y_pred_1' parameter must be an ndarray: got {type(y_pred_1)}"
                )
        
        if isinstance(y_pred_2, np.ndarray):
            self.y_pred_2 = y_pred_2
        elif isinstance(y_pred_2, pd.core.series.Series):
            self.y_pred_2 = y_pred_2.to_numpy()
        else:
            raise TypeError(
                f"'y_pred_2' parameter must be an ndarray: got {type(y_pred_2)}"
                )
        
        assert self.y_pred_1.shape[0] == self.y_pred_2.shape[0],\
        "'y_pred_1' and 'y_pred_2' parameters must have the same shape"
        
        if isinstance(exponential_coeff, float):
            self.exponential_coeff = exponential_coeff
        else:
            raise TypeError(
                f"'exponential_coeff' parameter must be a float: got {type(exponential_coeff)}"
                )
        
        if isinstance(threshold, float):
            self.threshold = threshold
        else:
            raise TypeError(
                f"'threshold' parameter must be a float: got {type(threshold)}"
                )
        
        # calibration of the axes to compute the exponential curve and display the graphs
        self.x_axis = np.arange(0, self.y_pred_1.shape[0])
        self.y_axis = (np.power(self.exponential_coeff, self.x_axis))
        
        # creation of a null vector which will be filled with the new predictions
        self.y_pred = np.zeros(shape=self.y_pred_1.shape[0])
        
        # check is fitted validation
        self.fitted = False
    
    
    def fit(self) -> None:
        """
        Function that allows to fit the joinded model.

        Returns
        -------
        None
            NoneType.

        """
        # calculation of the new predictions weighted by the exponential coefficient
        for idx in range(0, self.y_pred_1.shape[0]):
            alpha = self.y_axis[idx]
            self.y_pred[idx] = ((alpha*self.y_pred_1[idx])+((1-alpha)*self.y_pred_2[idx]))
        
        self.fitted = True
    
    
    def get_predictions(self) -> np.ndarray:
        """
        Function that allows to return new predictions from the joined model.

        Raises
        ------
        ValueError
            Before using this function, the joined model must be fitted.

        Returns
        -------
        self.y_pred : numpy.ndarray
            New predictions from the joined model.

        """
        if self.fitted:
            for idx in range(0, self.y_pred_1.shape[0]):
                self.y_pred[idx] = np.where(self.y_pred[idx] <= self.threshold, 0, 1)
            
            return self.y_pred
        else:
            raise ValueError(
                "Estimator's instance is not fitted yet. Call '.fit()' before using this function"
            )
    
    
    def get_plot(self, date=None, figsize: tuple=(30, 8),
                 colors: list=["black", "red", "blue", "green"],
                 title: str="New predictions", xlabel: str="Frequency",
                 ylabel: str="Value", label1: str="1st predictions",
                 label2: str="2nd predictions", xticks_rotation: int=0) -> None:
        """
        Function that allows to display a subplots with the exponential smoothing
        curve and the different predictions (y_pred_1, y_pred_2 and the news).

        Parameters
        ----------
        date : None or pandas.core.series.Series, optional, default=None
            If date parameter is specified, it will be added as an index for
            the x-axis of the plots.

        figsize : tuple, optional, default=(30, 8)
            Size of the subplots in pixels. Default is (width=30px, height=8px).

        colors : list, optional, default=["black", "red", "blue", "green"]
            Color list for subplots. Default is ["black", "red", "blue", "green"].

        title : str, optional, default="New predictions"
            Subplot's title. Default is "New predictions".

        xlabel : str, optional, default="Frequency"
            X-axis title. Default is "Frequency".

        ylabel : str, optional, default="Value"
            Y-axis title. Default is "Value".
        
        label1 : str, optional, default="1st predictions"
            Title for y_pred_1 values. Default is "1st predictions".
            
        label2 : str, optional, default="2nd predictions"
            Title for y_pred_2 values. Default is "2nd predictions".

        xticks_rotation : int, optional, default=0
            Labels' orientation on the x-axis. Default is 0.

        Raises
        ------
        TypeError
            - date parameter must be None or a series to use get_plot function.
            
            - figsize parameter must be a tuple to use get_plot function.
            
            - colors parameter must be a list to use get_plot function.
            
            - title parameter must be a str to use get_plot function.
            
            - xlabel parameter must be a str to use get_plot function.
            
            - ylabel parameter must be a str to use get_plot function.
            
            - label1 parameter must be a str to use get_plot function.
            
            - label2 parameter must be a str to use get_plot function.
            
            - xticks_rotation parameter must be an int to use get_plot function.

        ValueError
            Before using this function, the joined model must be fitted.

        AssertionError
            - date parameter must have the same shape as y_pred_1 and y_pred_2
            to use get_plot function.
            
            - figsize parameter must contain only the height and width to use
            get_plot function.
            
            - colors parameter must contain only 4 colors to use get_plot function.

        Returns
        -------
        None
            NoneType.

        """
        if date is None or isinstance(date, pd.core.series.Series):
            assert self.y_pred_1.shape[0] == date.shape[0],\
            "'date' parameter must have the same shape as 'y_pred_1' and 'y_pred_2'"
        else:
            raise TypeError(
                f"'date' parameter must be None or a series: got {type(date)}"
            )
        
        if isinstance(figsize, tuple):
            assert len(figsize) == 2,\
            "'figsize' parameter must contain the height and width: figsize=(width, height)"
        else:
            raise TypeError(
                f"'figsize' parameter must be a tuple: got {type(figsize)}"
            )
        
        if isinstance(colors, list):
            assert len(colors) == 4, "'colors' parameter must contain 4 colors"
        else:
            raise TypeError(
                f"'colors' parameter must be a list: got {type(colors)}"
            )
        
        if isinstance(title, str):
            pass
        else:
            raise TypeError(
                f"'title' parameter must be a str: got {type(title)}"
            )
        
        if isinstance(xlabel, str):
            pass
        else:
            raise TypeError(
                f"'xlabel' parameter must be a str: got {type(xlabel)}"
            )
        
        if isinstance(ylabel, str):
            pass
        else:
            raise TypeError(
                f"'ylabel' parameter must a str: got {type(ylabel)}"
            )
        
        if isinstance(label1, str):
            pass
        else:
            raise TypeError(
                f"'label1' parameter must a str: got {type(label1)}"
            )
        
        if isinstance(label2, str):
            pass
        else:
            raise TypeError(
                f"'label2' parameter must be a str: got {type(label2)}"
            )
        
        if isinstance(xticks_rotation, int):
            pass
        else:
            raise TypeError(
                f"'xticks_rotation' parameter must be an int: got {type(xticks_rotation)}"
            )
        
        if self.fitted:
            if date is None:
                df_values = pd.concat(
                    [
                        pd.DataFrame(self.y_pred_1, columns=[label1]),
                        pd.DataFrame(self.y_pred_2, columns=[label2])
                    ],
                    axis=1
                )
                df_values = pd.concat(
                    [
                        df_values,
                        pd.DataFrame(self.y_pred, columns=["New predictions"])
                    ],
                    axis=1
                )
                df_values.reset_index(drop=True, inplace=True)
            else:
                self.x_axis = date.to_frame(name="Date")
                df_values = pd.concat(
                    [
                        date.to_frame(name="Date"),
                        pd.DataFrame(self.y_pred_1, columns=[label1])
                    ],
                    axis=1
                )
                df_values = pd.concat(
                    [
                        df_values,
                        pd.DataFrame(self.y_pred_2, columns=[label2])
                    ],
                    axis=1
                )
                df_values = pd.concat(
                    [
                        df_values,
                        pd.DataFrame(self.y_pred, columns=["New predictions"])
                    ],
                    axis=1
                )
                df_values.reset_index(drop=True, inplace=True)
                df_values.set_index(keys="Date", drop=True, inplace=True)
            
            fig = plt.figure(figsize=figsize)
            
            plt.subplot(1, 2, 1)
            plt.plot(self.x_axis, self.y_axis, color=colors[0], label="Exponential smoothing")
            plt.title(f"Exponential smoothing curve for {self.exponential_coeff} coefficient")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend(loc="best")
            plt.xticks(rotation=xticks_rotation)
            
            plt.subplot(1, 2, 2)
            df_values[label1].plot(color=colors[1], label=label1)
            df_values[label2].plot(color=colors[2], label=label2)
            df_values["New predictions"].plot(color=colors[3], label="New predictions")
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend(loc="best")
            plt.xticks(rotation=xticks_rotation)
            
            plt.subplots_adjust(hspace=0.3)
            plt.show()
        else:
            raise ValueError(
                "Estimator's instance is not fitted yet. Call '.fit()' before using this function"
            )