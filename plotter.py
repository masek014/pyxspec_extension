import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib.ticker import MaxNLocator, AutoMinorLocator, ScalarFormatter, LogLocator, NullFormatter

from . import interface


FPM_COLORS = {
    'FPM A': 'black',
    'FPM B': 'red'
}


def compute_durbin_watson_statistic(residuals: np.ndarray) -> float:

    # Only use data up until the first inf appears.
    # Infs are caused by data points with error = 0.
    ind = np.argmin(np.abs(residuals) != np.inf)
    if ind != 0:
        residuals = residuals[:ind]

    denominator = np.sum(residuals[1:]**2)
    numerator = np.sum( (residuals[2:] - residuals[1:-1]) ** 2 )
    d = numerator / denominator

    return d


class ModelPlotter():
    

    def __init__(self, archive: 'interface.Archive'):
        self.archive = archive


    def get_model_instruments(self, model: str) -> list[str]:
        
        instruments = []
        for instrument, models in self.archive.instruments.items():
            if model in models:
                instruments.append(instrument)
        
        return instruments


    # TODO: implement this in the plot_data, plot_model, and plot_residual methods.
    def _initialize_plot(self, model: str, b_set_ticks: bool = True, **kwargs):

        plt.style.use(f'{os.path.dirname(__file__)}/styles/model.mplstyle')
        fig, ax = plt.subplots(**kwargs)

        for instrument in self.get_model_instruments(model):
            try:
                arrays = model[instrument][model].arrays
            except KeyError as e:
                print(e)
                continue
            color = FPM_COLORS[instrument]
            if 'energy_err' in arrays._fields:
                ax.errorbar(
                    x=arrays.energy.value,
                    y=arrays.data.value,
                    xerr=arrays.energy_err.value,
                    yerr=arrays.data_err.value,
                    ls='None',
                    color=color,
                    label=f'{instrument}'
                )
            if 'model' in arrays._fields:
                ax.plot(
                    arrays.energy.value,
                    arrays.model.value,
                    color=color,
                    drawstyle='steps-mid'
                )
         
            ax.set(
                xlabel=f'Energy ({arrays.energy.unit})',
                ylabel=f'{arrays.data.unit}',
                title=f'{model.capitalize()} model'
            )

        if b_set_ticks:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        return fig, ax


    def get_energy_limits(self, model: str) -> tuple:

        # TODO: I think we can use the min and max from the edges data.
        # TODO: Also, generalize this. Remove FPM A and FPM B.
        if len(self.get_model_instruments(model)) == 1:
            fpm = self.get_model_instruments(model)[0]
            arrays = self.archive.instruments[fpm][model].arrays
            energy = arrays.energy
            energy_err = arrays.energy_err
            xlim = (
                np.min( energy.value - energy_err.value ),
                np.max( energy.value + energy_err.value )
            ) * energy.unit
        else:
            arraysA = self.archive.instruments['FPM A'][model].arrays
            energyA = arraysA.energy
            energy_errA = arraysA.energy_err
            arraysB = self.archive.instruments['FPM B'][model].arrays
            energyB = arraysB.energy
            energy_errB = arraysB.energy_err
            
            minA = np.min( energyA.value - energy_errA.value )
            maxA = np.max( energyA.value + energy_errA.value )
            minB = np.min( energyB.value - energy_errB.value)
            maxB = np.max( energyB.value + energy_errB.value)

            xlim = (
                np.min([minA, minB]),
                np.max([maxA, maxB])
            ) * energyA.unit

        return xlim


    def plot_data(
        self,
        model: str,
        ax: matplotlib.axes.Axes = None,
        fpm: str = 'both',
        **kwargs
    ):
        """
        Plots the measured data that XSPEC fit models to.
        """

        if ax is None:
            plt.style.use(f'{os.path.dirname(__file__)}/styles/model.mplstyle')
            fig, ax = plt.subplots(sharex=True)

        if fpm == 'both':
            fpms = self.get_model_instruments(model)
        else:
            fpms = [fpm]

        for fpm in fpms:
            default_kwargs = dict(
                ls = 'None',
                color = FPM_COLORS[fpm]
            )
            fpm_kwargs = {**default_kwargs, **kwargs}

            arrays = self.archive.instruments[fpm][model].arrays
            energy = arrays.energy.value
            energy_err = arrays.energy_err.value
            data_arr = arrays.data.value
            data_err = arrays.data_err.value

            # Plot the data.
            ax.errorbar(
                energy, data_arr,
                xerr=energy_err, yerr=data_err,
                label=f'{fpm} data',
                **fpm_kwargs
            )
            ax.set(ylabel=f'Spectrum [{arrays.data.unit}]')

        return ax


    def plot_model(
        self,
        model: str,
        b_show_components: bool = False,
        ax: plt.Axes = None,
        fpm: str = 'both',
        model_kwargs: dict = {},
        component_kwargs: dict = {}
    ):

        if ax is None:
            plt.style.use(f'{os.path.dirname(__file__)}/styles/model.mplstyle')
            fig, ax = plt.subplots(sharex=True)

        if fpm == 'both':
            fpms = self.get_model_instruments(model)
        else:
            fpms = [fpm]

        for fpm in fpms:

            default_kwargs = dict(
                color = FPM_COLORS[fpm],
                label = f'{fpm} XSPEC model'
            )
            fpm_model_kwargs = {**default_kwargs, **model_kwargs}

            arrays = self.archive.instruments[fpm][model].arrays
            edges = arrays.energy_edges.value
            energy = arrays.energy.value
            model_arr = arrays.model.value

            # Plot the model.
            ax.stairs(
                model_arr, edges,
                **fpm_model_kwargs
            )
            ax.set(ylabel=f'Spectrum [{arrays.data.unit}]')

            # Add model components if they are available.
            if b_show_components:
                default_kwargs = dict(
                    ls = 'dotted',
                    # lw = 1.5,
                    alpha = 0.75,
                    color = FPM_COLORS[fpm]
                )
                fpm_component_kwargs = {**default_kwargs, **component_kwargs}
                for component_name, component in arrays.components.items():
                    ax.plot(
                        energy, component,
                        **fpm_component_kwargs
                    )

        return ax


    def plot_residuals(
        self,
        model: str,
        ax: plt.Axes = None,
        add_dw_stat: bool = False,
        **kwargs
    ):

        if ax is None:
            plt.style.use(f'{os.path.dirname(__file__)}/styles/model.mplstyle')
            fig, ax = plt.subplots(sharex=True)

        d_str = ''
        for fpm in self.get_model_instruments(model):

            default_kwargs = dict(
                ls = 'None',
                color = FPM_COLORS[fpm]
            )
            fpm_kwargs = {**default_kwargs, **kwargs}

            arrays = self.archive.instruments[fpm][model].arrays
            edges = arrays.energy_edges.value
            energy = arrays.energy.value
            energy_err = arrays.energy_err.value
            data_arr = arrays.data.value
            data_err = arrays.data_err.value
            model_arr = arrays.model.value

            residual = (data_arr - model_arr) / data_err
            d = compute_durbin_watson_statistic(residual)
            fpm_kwargs['label'] = rf'$d_{fpm[-1]}=${d:.3f}'

            ax.errorbar(
                energy, residual,
                xerr=energy_err,
                yerr=1,
                **fpm_kwargs
            )
            ax.set(ylabel='(data-model)/err')
            
        # if add_dw_stat:
        #     ax.text(
        #         # 0.95, 0.7,
        #         0.05, 0,
        #         d_str,
        #         ha='left',
        #         transform=ax.transAxes
        #     )

        return ax


    def make_xspec_plot(
        self,
        model: str,
        axs: list[plt.Axes, plt.Axes] = None,
        b_show_components: bool = True,
        b_show_parameters: bool = True
    ) -> list[plt.Axes, plt.Axes]:
        """
        This tries to emulate the style of an XSPEC model with residuals plot.
        """

        if axs is None:
            plt.style.use(f'{os.path.dirname(__file__)}/styles/model.mplstyle')
            fig, axs = plt.subplots(
                2, 1,
                figsize=(5,7),
                sharex=True,
                layout='constrained',
                gridspec_kw=dict(height_ratios=[3,1],hspace=0)
            )

        # Plot the data.
        self.plot_data(model, axs[0])
        self.plot_model(model, b_show_components, axs[0])
        self.plot_residuals(model, axs[1])

        if b_show_parameters:
            axs[0].text(
                0.05, 0.05,
                self.archive.make_model_string(model),
                ha='left',
                transform=axs[0].transAxes
            )

        axs[0].set(
            yscale='log',
            title=f'{model.capitalize()} model'
        )

        # Configure x-ticks.
        axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[0].xaxis.set_major_formatter(ScalarFormatter())

        # Configure y-ticks.
        locmaj = LogLocator(base=10,numticks=12) 
        axs[0].yaxis.set_major_locator(locmaj)
        locmin = LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=10)
        axs[0].yaxis.set_minor_locator(locmin)
        axs[0].yaxis.set_minor_formatter(NullFormatter())

        axs[0].legend()

        # Remove duplicate tick labels on shared x-axis.
        plt.setp(axs[0].get_xticklabels(), visible=False)
        
        axs[1].hlines(0.0, *(self.get_energy_limits(model).value), color='grey', ls='dotted')
        axs[1].set(
            xlabel='Energy [keV]',
            ylabel='(data-model)/(data error)',
            # xlim=xlim,
            ylim=[-8, 8]
        )
        axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
        axs[1].xaxis.set_minor_locator(AutoMinorLocator(5))
        axs[1].yaxis.set_minor_locator(AutoMinorLocator(2))

        return axs