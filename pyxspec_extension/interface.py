'''

'''

import os
import pickle

import astropy.units as u
import numpy as np

from astropy.io import fits
from collections import OrderedDict, namedtuple
from dataclasses import dataclass, field

import xspec

from .config.configuration import XSPECConfiguration
from .custom_models import add_all_custom_models
# from .plotter import compute_durbin_watson_statistic


# TODO: Make the interface class use the ModelPlotter class?

add_all_custom_models()

CONSTANT_EXPRESSION_NO_PAREN = '{outer}constant*{inner}'
CONSTANT_EXPRESSION_PAREN = '{outer}constant*({inner})'
PARAMETER_LINK_FORMAT = '= {model_name}:p{param_num}'

EXPRESSIONS = dict(
    single='const * vapec',
    double='const * (vapec + vapec)',
    broken='const * (bknpower + vapec)'
)

MODEL_COMPONENT_MAPPING = dict(
    diagnostic=dict(
        EXPOSURE=('exposure', u.s),
        STATISTIC=('statistic', u.Unit()),
        DOF=('dof', u.Unit())
    ),
    constant=dict(
        factor=('factor', u.Unit())
    ),
    gaussian=dict(
        LineE=('$E_L$', u.keV),
        Sigma=('sigma', u.keV),
        norm=('gauss_norm', u.ph/(u.cm**2)/u.s)
    ),
    nlapec=dict(
        kT=('t', u.keV),
        norm=('em', u.Unit())
    ),
    vapec=dict(
        kT=('t', u.keV),
        norm=('em', u.Unit())
    ),
    bknpower=dict(
        PhoIndx1=('lower_index', u.Unit()),
        BreakE=('$E_B$', u.keV),
        PhoIndx2=('index', u.Unit()),
        norm=('norm', u.ph/u.keV/(u.cm**2)/u.s)
    ),
    expmodgauss=dict(  # TODO: FIGURE OUT UNITS!!!
        norm=('expmodgauss_norm', u.Unit()),
        lam=('lambda', u.ct / u.keV),
        mu=('mu', u.keV),
        sigma=('sigma', u.keV),
    )
)

# TODO: Combine with the dictionary above?
# TODO: This norm parameter conversion is specfically for Earth-based observers... Fix this.
PARAMETER_CONVERSIONS = dict(
    nlapec=dict(
        kT=11.6045 * u.MK / u.keV,
        norm=u.cm**-3 / 3.5557e-42
    ),
    vapec=dict(
        kT=11.6045 * u.MK / u.keV,
        norm=u.cm**-3 / 3.5557e-42
    )
)
COMPONENT_PARAMETER_MAPPING = dict(
    constant=dict(
        factor=('factor', u.Unit())
    ),
    gaussian=dict(
        LineE=('line_energy', u.keV),
        Sigma=('line_width', u.keV),
        norm=('gaussian_norm', u.ph/(u.cm**2)/u.s)
    ),
    nlapec=dict(
        kT=('temperature', u.keV),
        norm=('emission_measure', u.Unit())
    ),
    vapec=dict(
        kT=('temperature', u.keV),
        norm=('emission_measure', u.Unit())
    ),
    bknpower=dict(
        norm=('powerlaw_norm', u.ph/u.keV/(u.cm**2)/u.s),
        BreakE=('break_energy', u.keV),
        PhoIndx1=('lower_index', u.Unit()),
        PhoIndx2=('upper_index', u.Unit()),
    ),
    expmodgauss=dict(  # TODO: FIGURE OUT UNITS!!!
        norm=('expmodgauss_norm', u.Unit()),
        lam=('lambda', u.ct / u.keV),
        mu=('mu', u.keV),
        sigma=('sigma', u.keV),
    )
)


ModelResult = namedtuple(
    'ModelResult',
    [
        'energy_edges',
        'energy',
        'energy_err',
        'data',
        'data_err',
        'model',
        'components'
    ]
)


@dataclass
class config:
    # abundance_file: str = '/home/reed/feld92a_coronal0.txt'
    debug: bool = True


CONFIG = config()


# TODO: Put this in a different file. histogram_tools.py?
def compute_energy_edges(energy: np.ndarray, energy_err: np.ndarray) -> np.ndarray:
    return np.append(
        energy - energy_err,
        [energy[-1] + energy_err[-1]]
    )


def gather_model_data(model_name: str, data_group: int) -> ModelResult:

    # Configure the PlotManager so we can retrieve model and component spectra.
    xspec.Plot.add = True
    xspec.Plot.background = False
    xspec.Plot.xAxis = "keV"
    xspec.Plot("data")

    energy = np.array(xspec.Plot.x(data_group)) * u.keV
    energy_err = np.array(xspec.Plot.xErr(data_group)) * u.keV
    energy_edges = compute_energy_edges(energy, energy_err)
    result = ModelResult(
        energy_edges=energy_edges,
        energy=energy,
        energy_err=energy_err,
        data=np.array(xspec.Plot.y(data_group)) * u.ct/u.s/u.keV,
        data_err=np.array(xspec.Plot.yErr(data_group)) * u.ct/u.s/u.keV,
        model=np.array(xspec.Plot.model(data_group)) * u.ct/u.s/u.keV,
        components={}
    )

    # Add components.
    component_names = []
    for component_name in xspec.AllModels(data_group, model_name).componentNames:
        if 'constant' not in component_name:
            component_names.append(component_name)
    print('result component names:', component_names)
    if len(component_names) > 1:
        for component_num, component_name in enumerate(component_names, start=1):
            print('adding component to result:', component_name)
            comp = xspec.Plot.addComp(component_num, data_group)
            result.components[component_name] = np.array(comp) * u.ct/u.s/u.keV

    return result


class ArchivedParameter():

    def __init__(
        self,
        parameter: xspec.Parameter,
        label: str,
        unit: u.Unit
    ):
        self.name = parameter.name
        self.values = parameter.values
        self.error = parameter.error
        self.link = parameter.link
        self.frozen = parameter.frozen
        self.label = label
        self.unit = unit

    @property
    def quantity(self) -> u.Quantity:
        return self.value * self.unit

    @property
    def lower(self) -> u.Quantity:
        return self.error[0] * self.unit

    @property
    def upper(self) -> u.Quantity:
        return self.error[1] * self.unit

    @property
    def value(self) -> float:
        return self.values[0]

    @property
    def step(self) -> float:
        return self.values[1]

    @property
    def hard_min(self) -> float:
        return self.values[2]

    @property
    def soft_min(self) -> float:
        return self.values[3]

    @property
    def soft_max(self) -> float:
        return self.values[4]

    @property
    def hard_max(self) -> float:
        return self.values[5]


class ArchivedComponent():

    def __init__(self, component: xspec.Component):
        self.name = component.name
        self.parameter_names = component.parameterNames
        self.parameters = {}
        for parameter_name in self.parameter_names:
            component_parameter = component.__dict__[parameter_name]

            mapping = COMPONENT_PARAMETER_MAPPING[self.base_name]
            if parameter_name in mapping:
                label, unit = mapping[parameter_name]
            else:
                label = parameter_name
                unit = parameter.unit
                print(
                    f'Warning: parameter {parameter_name} not in COMPONENT_PARAMETER_MAPPING.')

            parameter = ArchivedParameter(
                component_parameter,
                label,
                unit
            )

            self.parameters[parameter_name] = parameter
            setattr(self, parameter_name, parameter)

    @property
    def base_name(self) -> str:
        '''The name of the component without any indices.'''
        return (self.name).split('_')[0]

    @property
    def free_parameters(self) -> list[ArchivedParameter]:
        '''All free parameters of the component.'''
        parameters = []
        for parameter_name in self.parameters:
            parameter = self.parameters[parameter_name]
            if not parameter.link and not parameter.frozen:
                parameters.append(parameter)

        return parameters

    @property
    def parameter_string(self) -> str:
        '''Formatted string containing parameter values.'''
        for parameter in self.parameters:
            value = parameter.quantity
            errors = np.array([*parameter.error[0:2]]) * parameter.unit
            errors -= value

            if self.base_name in PARAMETER_CONVERSIONS:
                conversions = PARAMETER_CONVERSIONS[self.base_name]
                if parameter.name in conversions:
                    value *= conversions[parameter.name]
                    errors *= conversions[parameter.name]

            # Scientific notation is used based on order of magnitude.
            if value.value >= 1000:
                str_format = '{:0.{prec}E}'
            else:
                str_format = '{:0.{prec}f}'

            # Determine prec, which represents the number of decimal places.
            mantissa = errors[1].value % 1
            prec = 1
            if mantissa != 0:
                while (mantissa * 10 ** prec) < 1 and prec < 4:
                    prec += 1

            value_str = str_format.format(value.value, prec=prec)
            error_low = str_format.format(errors[0].value, prec=prec)
            error_high = str_format.format(errors[1].value, prec=prec)
            error_str = rf'$_{{{error_low}}}^{{+{error_high}}}$'

            mapping = MODEL_COMPONENT_MAPPING[self.base_name]
            param_name = mapping[parameter.name][0]
            unit_str = f'[{value.unit}]'
            if value.unit == u.Unit():
                unit_str = ''

            full_string += f'{param_name.upper()}: {value_str}{error_str} {unit_str}\n'

        return full_string


# TODO: Add the model components and full model to this. (idr what this means)
class ArchivedModel():
    '''Stores model data that is to remain unchanged.'''

    def __init__(
        self,
        label: str,
        source: int,
        data_group: int,
        model: xspec.Model,
        data_file: str = None,
        out_dir: str = None,
    ):

        self.label = label
        self.source = source
        self.data_group = data_group
        self.name = model.name
        self.expression = model.expression
        self.statistic = xspec.AllData(data_group).statistic
        self.data_file = data_file
        self.out_dir = out_dir

        self._archive_components(model)
        self._archive_response()
        self._archive_arrays()

    @property
    def free_parameters(self) -> dict[str, list[ArchivedParameter]]:
        '''All free parameters associated with the model.'''
        parameters = {}
        for component_name in self.components:
            component = self.components[component_name]
            parameters[component] = component.free_parameters

        return parameters

    @property
    def free_response_parameters(self) -> list[ArchivedParameter]:
        '''All free *response* parameters (e.g. gain) associated with the model.'''
        rparameters = []
        for rparameter_name in self.response_parameters:
            rparameter = self.response_parameters[rparameter_name]
            if not rparameter.link and not rparameter.frozen:
                rparameters.append(rparameter)

        return rparameters

    @property
    def fit_statistic_string(self) -> str:
        return f'{self.label} CSTAT: {self.statistic:0.1f} ({self.bins} bins)'

    @property
    def parameter_string(self) -> str:

        base_component_names = [c.base_name for c in self.components.values()]
        unique = np.unique(base_component_names, return_counts=True)
        unique_components = {u: {'uses': 0, 'counts': c}
                             for u, c in zip(unique[0], unique[1])}

        full_string = ''
        component_parameters = self.free_parameters
        for component, parameters in component_parameters.items():

            postfix = ''
            if component.base_name in unique_components:
                counts = unique_components[component.base_name]['counts']
                if counts > 1:
                    unique_components[component.base_name]['uses'] += 1
                    postfix = rf'$_{unique_components[component.base_name]["uses"]}$'

            for parameter in parameters:

                value = parameter.quantity
                errors = np.array([*parameter.error[0:2]]) * parameter.unit
                errors -= value

                # TODO: should we make this use the parameter_string property from ArchivedComponent?
                if component.base_name in PARAMETER_CONVERSIONS:
                    conversions = PARAMETER_CONVERSIONS[component.base_name]
                    if parameter.name in conversions:
                        value *= conversions[parameter.name]
                        errors *= conversions[parameter.name]

                # Scientific notation is used based on order of magnitude.
                if value.value >= 1000:
                    str_format = '{:0.{prec}E}'
                else:
                    str_format = '{:0.{prec}f}'

                # Determine prec, which represents the number of decimal places.
                mantissa = errors[1].value % 1
                prec = 1
                if mantissa != 0:
                    while (mantissa * 10 ** prec) < 1 and prec < 4:
                        prec += 1

                value_str = str_format.format(value.value, prec=prec)
                error_low = str_format.format(errors[0].value, prec=prec)
                error_high = str_format.format(errors[1].value, prec=prec)
                error_str = rf'$_{{{error_low}}}^{{+{error_high}}}$'

                mapping = MODEL_COMPONENT_MAPPING[component.base_name]
                param_name = mapping[parameter.name][0]
                unit_str = f'[{value.unit}]'
                if value.unit == u.Unit():
                    unit_str = ''

                full_string += f'{param_name.upper()+postfix}: {value_str}{error_str} {unit_str}\n'

        return full_string

    @property
    def response_parameter_string(self) -> str:

        full_string = ''
        if self.response_parameters:
            # if not self.gain_slope.frozen and not self.gain_slope.link:
            if not self.gain_slope.link:
                full_string += f'GAIN SLOPE: {self.gain_slope.value:0.3f}'
            if not self.gain_offset.frozen and not self.gain_offset.link:
                full_string += f'\nGAIN OFFSET: {self.gain_offset.value:0.3f}'

        return full_string

    def _archive_components(self, model: xspec.Model):
        '''Archives the model components and their parameters.'''
        component_names = np.array([c.split('_')[0]
                                   for c in model.componentNames])
        names, counts = np.unique(component_names, return_counts=True)
        occurences = {n: {'counts': c, 'uses': 0}
                      for n, c in zip(names, counts)}
        self.components = {}
        for full_component_name in model.componentNames:
            component = model.__dict__[full_component_name]
            component = ArchivedComponent(model.__dict__[full_component_name])
            component_name = full_component_name.split('_')[0]
            if occurences[component_name]['counts'] > 1:
                occurences[component_name]["uses"] += 1
                component_name = f'{component_name}{occurences[component_name]["uses"]}'
            self.components[component_name] = component
            setattr(self, component_name, component)

    def _archive_response(self):
        '''Archives the response parameters, if active.'''
        self.response_parameters = {}
        gain = xspec.AllData(self.data_group).multiresponse[self.source-1].gain
        if gain.isOn:
            for parameter_name in gain.parameterNames:
                new_parameter_name = f'gain_{parameter_name}'
                parameter = ArchivedParameter(
                    gain.__dict__[parameter_name],
                    new_parameter_name,
                    u.Unit()
                )
                self.response_parameters[parameter_name] = parameter
                setattr(self, new_parameter_name, parameter)

    def _archive_arrays(self):
        '''Archives the data arrays, i.e. the data for plotting.'''
        self.arrays = gather_model_data(self.name, self.data_group)
        self.bins = int(len(self.arrays.energy))

    def _component_is_active(self, component: xspec.Component) -> bool:
        '''Returns if the component is active, i.e. not frozen.'''
        is_active = False
        parameter_names = component.parameterNames[:]
        while not is_active and parameter_names:
            name = parameter_names.pop()
            parameter = component.__dict__[name]
            is_active = not parameter.frozen

        return is_active


@dataclass
class Archive():
    instruments: dict[str, OrderedDict[str, ArchivedModel]
                      ] = field(default_factory=dict)

    def last_instrument_model(self, instrument: str) -> ArchivedModel:
        last = next(reversed(self.instruments[instrument]))
        return self.instruments[instrument][last]

    def add_model(self, instrument: str, model: ArchivedModel, model_name: str | None = None):

        if instrument not in self.instruments:
            self.instruments[instrument] = OrderedDict()

        if model_name is None:
            model_name = model.name
        if model_name not in self.instruments[instrument]:
            self.instruments[instrument][model_name] = model
            # This is the main reason I want this class
            setattr(self.instruments[instrument], model_name, model)
        else:
            print(
                f'WARNING: model \'{model_name}\' already in Archive for instrument {instrument}. Not doing anything.')

    def save(self, pickle_path: str):
        '''Saves self to the provided pickle path. This is intended to be used
        in conjunction with the 'load' class method.
        '''
        with open(pickle_path, 'wb') as outfile:
            pickle.dump(self, outfile, 2)

    @classmethod
    def load(cls, pickle_path: str):
        '''Creates an instance of the Archive class from the provided pickle.'''
        with open(pickle_path, 'rb') as infile:
            return pickle.load(infile)

    def make_model_string(self, model: str) -> str:

        param_strs, fit_stat_strs, response_strs = [], [], []
        for models in self.instruments.values():
            if model not in models:
                continue
            archived_model = models[model]
            param_strs.append(archived_model.parameter_string)
            fit_stat_strs.append(archived_model.fit_statistic_string)
            response_str = archived_model.response_parameter_string
            if response_str:
                response_strs.append(response_str)

        full_str = ''.join(param_strs)
        full_str += '\n'.join(fit_stat_strs) + '\n'
        full_str += '\n'.join(response_strs)

        return full_str

    def make_multimodel_string(self, models: list[str]) -> str:

        full_str = ''
        for model in models:
            full_str += f'{model.upper()}:\n{self.make_model_string(model)}\n'

        return full_str


@dataclass
class Instrument:
    name: str
    signal_file: str
    signal_data_group: int
    signal_source: int
    pileup_file: str = None
    pileup_data_group: int = None
    pileup_source: int = None
    background_file: str = None
    background_data_group: int = None
    background_source: int = None

    # TODO: Better way of getting these keywords?
    @property
    def signal_response_file(self) -> str:
        with fits.open(self.signal_file) as hdu:
            respfile = hdu[1].header['RESPFILE']
        return respfile

    @property
    def pileup_response_file(self) -> str | None:
        if self.pileup_file is not None:
            with fits.open(self.pileup_file) as hdu:
                respfile = hdu[1].header['RESPFILE']
            return respfile

    @property
    def background_response_file(self) -> str | None:
        if self.background_file is not None:
            with fits.open(self.background_file) as hdu:
                respfile = hdu[1].header['RESPFILE']
            return respfile

    @property
    def signal_spectrum(self) -> xspec.Spectrum:
        return xspec.AllData(self.signal_data_group)

    @property
    def pileup_spectrum(self) -> xspec.Spectrum:
        if self.pileup_data_group is not None:
            return xspec.AllData(self.pileup_data_group)

    @property
    def background_spectrum(self) -> xspec.Spectrum:
        if self.background_data_group is not None:
            return xspec.AllData(self.background_data_group)

    @property
    def pileup_model_name(self) -> str | None:
        if self.pileup_file is not None:
            return f'pileup{self.name}'.replace(' ', '')

    @property
    def background_model_name(self) -> str | None:
        if self.background_file is not None:
            return f'background'.replace(' ', '')

    def get_signal_model(self, model_name: str) -> xspec.Model:
        return xspec.AllModels(self.signal_data_group, model_name)

    def get_pileup_model(self) -> xspec.Model | None:
        if self.pileup_file is not None:
            return xspec.AllModels(self.pileup_data_group, self.pileup_model_name)

    def get_background_model(self) -> xspec.Model | None:
        if self.background_file is not None:
            return xspec.AllModels(self.background_data_group, self.background_model_name)

    # TODO: Add methods/properties to get the pileup and background components in the signal.


class XSPECInterface:
    '''This interface has been tested with NuSTAR inputs, but MIGHT work
    for other instruments provided that the input files obey the OGIP
    standard.
    '''

    def __init__(self):
        self.archive = Archive()
        self.instruments = OrderedDict()
        self.signal_source = 1
        self.signal_groups = []
        self.pileup_sources = OrderedDict()
        self.pileup_groups = []
        self.pileup_expression = ''
        self.background_sources = OrderedDict()
        self.background_groups = []
        self.background_expression = ''

    @property
    def num_groups(self) -> int:
        '''Number of data groups being used.'''
        return len(self.signal_groups) + len(self.pileup_groups) + len(self.background_groups)

    def add_instrument(
        self,
        name: str,
        signal_file: str,
        pileup_file: str = None,
        background_file: str = None,
        link_background: bool = False
    ):
        '''TODO: if we wanted to allow separate pileup models for each
        instrument, we would need to set different signal sources
        for each instrument. Right now, this cannot be done.

        TODO: implement link_background
        '''

        group_num = self.num_groups + 1
        self.signal_groups.append(group_num)
        signal_kwargs = dict(
            signal_file=signal_file,
            signal_data_group=group_num,
            signal_source=1
        )

        pileup_kwargs = {}
        if pileup_file is not None:
            group_num = self.num_groups + 1
            source_num = len(self.pileup_sources) + \
                len(self.background_sources) + 2
            self.pileup_sources[source_num] = name
            self.pileup_groups.append(group_num)
            pileup_kwargs = dict(
                pileup_file=pileup_file,
                pileup_data_group=group_num,
                pileup_source=source_num
            )

        background_kwargs = {}
        if background_file is not None:
            group_num = self.num_groups + 1

            # Make all backgrounds the same model... need to change this.
            if len(self.background_sources) > 0:
                source_num = list(self.background_sources.keys())[0]
            else:
                source_num = len(self.pileup_sources) + \
                    len(self.background_sources) + 2
            self.background_sources[source_num] = name
            self.background_groups.append(group_num)
            background_kwargs = dict(
                background_file=background_file,
                background_data_group=group_num,
                background_source=source_num
            )

        self.instruments[name] = Instrument(
            name,
            **signal_kwargs,
            **pileup_kwargs,
            **background_kwargs
        )
        print('Instrument:', name)
        print('\t', signal_kwargs)
        print('\t', pileup_kwargs)
        print('\t', background_kwargs)

    def _configure_responses(self):

        # Check if ANY instruments have pileup files.
        # If there's at least one, then configure the multiresponse
        # for each source for each Spectrum object.
        pileup = False
        for instrument in self.instruments.values():
            if instrument.pileup_file is not None:
                pileup = True
                break

        background = False
        for instrument in self.instruments.values():
            if instrument.background_file is not None:
                background = True
                break

        num_sources = len(self.pileup_sources) + \
            len(self.background_sources) + 1
        # TODO: could consolidate this?
        if pileup:
            for instrument in self.instruments.values():
                signal_spectrum = instrument.signal_spectrum
                signal_spectrum.multiresponse[0] = instrument.signal_response_file
                for source_num in range(1, num_sources):
                    signal_spectrum.multiresponse[source_num] = 'none'
                    print(
                        f'{instrument.name} signal_spectrum.multiresponse[{source_num}] = "none"')

                pileup_spectrum = instrument.pileup_spectrum
                if pileup_spectrum is not None:
                    for source_num in range(0, num_sources):
                        if source_num != instrument.pileup_source - 1:
                            pileup_spectrum.multiresponse[source_num] = 'none'
                            print(
                                f'{instrument.name} pileup_spectrum.multiresponse[{source_num}] = "none"')
                        else:
                            pileup_spectrum.multiresponse[source_num] = instrument.pileup_response_file
                            print(
                                f'{instrument.name} pileup_spectrum.multiresponse[{source_num}] = {instrument.pileup_response_file}')

        if background:
            for instrument in self.instruments.values():
                signal_spectrum = instrument.signal_spectrum
                signal_spectrum.multiresponse[0] = instrument.signal_response_file
                for source_num in range(1, num_sources):
                    signal_spectrum.multiresponse[source_num] = 'none'
                    print(
                        f'{instrument.name} signal_spectrum.multiresponse[{source_num}] = "none"')

                background_spectrum = instrument.background_spectrum
                if background_spectrum is not None:
                    for source_num in range(0, num_sources):
                        if source_num != instrument.background_source - 1:
                            background_spectrum.multiresponse[source_num] = 'none'
                            print(
                                f'{instrument.name} background_spectrum.multiresponse[{source_num}] = "none"')
                        else:
                            background_spectrum.multiresponse[source_num] = instrument.background_response_file
                            print(
                                f'{instrument.name} background_spectrum.multiresponse[{source_num}] = {instrument.background_response_file}')

    def clear_data(self):

        xspec.AllData.clear()
        self.signal_groups = []
        self.pileup_groups = []
        self.pileup_sources = OrderedDict()
        self.background_groups = []
        self.background_sources = OrderedDict()

    def read_data(self, data_dir: str):
        '''data_dir should contain all of the relevant PHA files and
        corresponding RMF files and ARF files.

        XSPEC assumes that the data files referenced within FITS headers (e.g.
        the RESPFILE for the RMF) are contained within the directory
        specified within the header. If there is no directory specified, i.e.
        only the file name is specified, then XSPEC will search the directory
        from which XSPEC is executed.
        '''

        data_str = ''
        for instrument in self.instruments.values():
            data_str += f'{instrument.signal_data_group}:{instrument.signal_data_group} {instrument.signal_file} '
            if instrument.pileup_file is not None:
                data_str += f'{instrument.pileup_data_group}:{instrument.pileup_data_group} {instrument.pileup_file} '
            if instrument.background_file is not None:
                data_str += f'{instrument.background_data_group}:{instrument.background_data_group} {instrument.background_file} '

        orig_dir = os.getcwd()
        os.chdir(data_dir)
        print(data_str)
        xspec.AllData(data_str)
        self._configure_responses()
        os.chdir(orig_dir)

    def archive_previous(self) -> OrderedDict[str, dict[str, ArchivedModel]]:
        '''Archives the currently loaded Model.
        TODO: Remove return?
        TODO: There's a bug where, if archive_previous is called multiple times, the same model get applied to all instruments.
        '''

        newly_archived = OrderedDict()
        if not hasattr(self, 'current_model') or not hasattr(self, 'results'):
            return newly_archived

        for instrument in self.instruments.values():

            try:
                newly_archived[instrument.name] = {}
                instrument.get_signal_model(self.current_model)
            except Exception as e:
                continue

            if instrument.background_file is not None:
                archived_background_model = ArchivedModel(
                    instrument.name,
                    instrument.background_source,
                    instrument.background_data_group,
                    instrument.get_background_model(),
                    instrument.background_file,
                    self.out_dir
                )
                self.archive.add_model(
                    instrument.name, archived_background_model)
                newly_archived[instrument.name]['background'] = archived_background_model

            if instrument.pileup_file is not None:
                archived_pileup_model = ArchivedModel(
                    instrument.name,
                    instrument.pileup_source,
                    instrument.pileup_data_group,
                    instrument.get_pileup_model(),
                    instrument.pileup_file,
                    self.out_dir
                )
                # I'm not a fan of combining the pileup model and current model here.
                # TODO: try to incorporate this elsewhere.
                self.archive.add_model(instrument.name, archived_pileup_model,
                                       model_name=f'{archived_pileup_model.name}_{self.current_model}')
                newly_archived[instrument.name]['pileup'] = archived_pileup_model

            archived_signal_model = ArchivedModel(
                instrument.name,
                instrument.signal_source,
                instrument.signal_data_group,
                instrument.get_signal_model(self.current_model),
                instrument.signal_file,
                self.out_dir
            )
            self.archive.add_model(instrument.name, archived_signal_model)
            newly_archived[instrument.name]['signal'] = archived_signal_model

        print('newly_archived:', newly_archived)
        if CONFIG.debug:
            for instrument, models in newly_archived.items():
                for name, model in models.items():
                    print(
                        f'Archived model {instrument} {model.name}, expression {model.expression}')
                    for component_name, component in model.components.items():
                        print(f'\tComponent {component_name}')
                        for parameter_name, parameter in component.parameters.items():
                            print(f'\t\tParameter {parameter_name}')
                            print(f'\t\t\tValues: {parameter.values}')
                            print(f'\t\t\tFrozen: {parameter.frozen}')

        return newly_archived

    def _configure_new_expression(self, new: str, old: str) -> str:

        if 'const*' in new:
            new = new.replace('const*', 'constant*')

        if self.pileup_expression:
            new = f'{self.pileup_expression} + {new}'

        if self.background_expression:
            new = f'{self.background_expression} + {new}'

        if old:
            # if 'constant' in old:
            #     old = f' {old}' # Prefix space to work with formats
            #     p = parse.parse(CONSTANT_EXPRESSION_PAREN, old)
            #     if p is None:
            #         p = parse.parse(CONSTANT_EXPRESSION_NO_PAREN, old)
            #     p.named['inner'] = f'{p.named["inner"]} + {new}'
            #     new = CONSTANT_EXPRESSION_PAREN.format(**p.named)
            # else:
            new = f'{old} + {new}'

        return new

    def set_pileup_model(self, expression: str) -> list[xspec.Model]:
        '''Applies the given expression to ALL instrument pileup models.
        Currently, individual instrument pileup models cannot be set
        due to how the sources are handled.

        This model is fit simultaneous to the other models.
        '''

        models = []
        self.pileup_expression = expression
        for source, instrument_name in self.pileup_sources.items():
            instrument = self.instruments[instrument_name]
            print(expression, instrument.pileup_model_name, source)
            model = xspec.Model(
                expression, instrument.pileup_model_name, source)
            models.append(model)

        return models

    def set_background_model(self, expression: str) -> list[xspec.Model]:
        '''Applies the given expression to ALL instrument background models.
        Currently, individual instrument background models cannot be set
        due to how the sources are handled.

        This model is fit simultaneous to the other models.
        '''

        models = []
        self.background_expression = expression
        for source, instrument_name in self.background_sources.items():
            instrument = self.instruments[instrument_name]
            model = xspec.Model(
                expression, instrument.background_model_name, source)
            models.append(model)

        return models

    def _set_pileup_links(self):
        '''Links the parameters signal's pileup model component(s) parameters
        to the corresponding parameters of the pileup model.
        '''

        no_pileup_instruments = []
        for instrument in self.instruments.values():
            signal_model = instrument.get_signal_model(self.current_model)
            pileup_model = instrument.get_pileup_model()
            if pileup_model is not None:
                ref_model = pileup_model
                for component_name in pileup_model.componentNames:
                    component = pileup_model.__dict__[component_name]
                    signal_component_name = instrument.pileup_components_in_signal[component_name]
                    for parameter_name in component.parameterNames:
                        parameter = component.__dict__[parameter_name]
                        signal_model.__dict__[signal_component_name].__dict__[
                            parameter_name].link = parameter
            else:
                no_pileup_instruments.append(instrument)

        # Set all signal parameters corresponding to the pileup model to zero.
        # TODO: See if we can do this better.
        for instrument in no_pileup_instruments:
            signal_model = instrument.get_signal_model(self.current_model)
            pileup_component_names = ref_model.componentNames
            for component_name in pileup_component_names:
                signal_component_name = instrument.pileup_components_in_signal[component_name]
                component = signal_model.__dict__[signal_component_name]
                for parameter_name in component.parameterNames:
                    parameter = component.__dict__[parameter_name]
                    parameter.values = '0 0 0 0 0 0'
                    parameter.frozen = True

    def _set_background_links(self):
        '''Links the parameters signal's background model component(s) parameters
        to the corresponding parameters of the background model.
        '''

        no_background_instruments = []
        for instrument in self.instruments.values():
            signal_model = instrument.get_signal_model(self.current_model)
            background_model = instrument.get_background_model()
            if background_model is not None:
                ref_model = background_model
                for component_name in background_model.componentNames:
                    component = background_model.__dict__[component_name]
                    signal_component_name = instrument.background_components_in_signal[
                        component_name]
                    for parameter_name in component.parameterNames:
                        parameter = component.__dict__[parameter_name]
                        signal_model.__dict__[signal_component_name].__dict__[
                            parameter_name].link = parameter
            else:
                no_background_instruments.append(instrument)
        print('ref_model:', ref_model)

        # Set all signal parameters corresponding to the background model to zero.
        # TODO: See if we can do this better.
        for instrument in no_background_instruments:
            signal_model = instrument.get_signal_model(self.current_model)
            background_component_names = ref_model.componentNames
            for component_name in background_component_names:
                signal_component_name = instrument.background_components_in_signal[component_name]
                component = signal_model.__dict__[signal_component_name]
                for parameter_name in component.parameterNames:
                    parameter = component.__dict__[parameter_name]
                    parameter.values = '0 0 0 0 0 0'
                    parameter.frozen = True

    def add_component(
        self,
        model_name: str,
        expression: str,
        parameter_limits_file: str,
        out_dir: str,
        freeze_previous: bool = False,
        tie_data_groups: bool = True,
    ) -> xspec.Model:
        '''Adds a new expression to the overall model.
        Allows adding new components to an already existing model.

        TODO: this is a mess. Need to clean up.
        '''

        self.out_dir = out_dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir, exist_ok=True)

        # Setup log file. Each model name combination gets its own log.
        xspec.Xset.chatter = 0
        self.logfile = xspec.Xset.openLog(
            f'{self.out_dir}/model_{model_name}.log')
        # self.archive_previous()

        self.current_model = model_name

        # Get the most recent model(s) archived.
        # It is done here so that the user can call 'archive_previous()'
        # whenever they want, but 'add_component()' is not dependent on
        # the user doing so.
        archived_models = {}
        if self.archive.instruments:
            for instrument in self.archive.instruments:
                model_name = next(
                    reversed(self.archive.instruments[instrument]))
                archived_models[instrument] = self.archive.instruments[instrument][model_name]
                print('Recently archived model:', instrument, model_name)
            old_expression = archived_models[instrument].expression
        else:
            old_expression = ''

        # Set up the new model.
        new_expression = self._configure_new_expression(
            expression, old_expression)
        print(
            f'Adding new component {self.current_model} with expression {expression}')
        print('Full expression is now', new_expression)
        model = xspec.Model(
            new_expression, self.current_model, self.signal_source)
        print(model.name)
        # TODO: Create some way of setting pileup parameter limits?
        print('parameter limits file:', parameter_limits_file)
        self._set_parameter_limits(parameter_limits_file, tie_data_groups)

        all_components = [c for c in model.componentNames]
        print('all_components:', all_components)

        # Add the archived components to the new model.
        if archived_models:
            archived_models_it = iter(archived_models)
            for group_num in self.signal_groups:
                current_model = xspec.AllModels(group_num, self.current_model)
                archived_model = archived_models[next(archived_models_it)]
                for old_component in archived_model.components.values():
                    # if old_component.name == 'constant' and group_num == 1:
                    #     continue
                    new_component = current_model.__dict__[old_component.name]
                    if old_component.name in all_components:
                        all_components.remove(old_component.name)
                        print(
                            f'removing {old_component.name} from all_components')
                    print(
                        f'reading parameters from component {old_component.name} of archived model', archived_model.name)
                    for parameter_name in new_component.parameterNames:
                        print(parameter_name)
                        if parameter_name in old_component.parameters:
                            old_parameter = old_component.parameters[parameter_name]
                            print(
                                f'parameter {parameter_name} linked to {old_parameter.link}')
                            parameter = new_component.__dict__[parameter_name]
                            # if old_parameter.link:
                            #     print(f'old parameter {parameter_name} in old model is linked. skipping')
                            #     continue
                            parameter.values = old_parameter.values
                            # parameter.error = old_parameter.error # TODO: Figure out how to set this.
                            parameter.frozen = old_parameter.frozen or freeze_previous
                            print(
                                f'parameter {parameter_name} frozen:', old_parameter.frozen)
                            print('freeze previous:', freeze_previous)
                            print(
                                f'applying {parameter_name} values to component {new_component.name} of new model {model.name}. frozen: {parameter.frozen}, {parameter.values}')

        # all_components is now only the new components in the signal model
        to_remove = []
        for instrument in self.instruments.values():
            instrument.pileup_components_in_signal = {}
            for component_name in all_components:
                if instrument.get_pileup_model() is not None:
                    for pileup_component in instrument.get_pileup_model().componentNames:
                        print(
                            f'pileup_component {pileup_component} in {instrument.name}')
                        if pileup_component not in instrument.pileup_components_in_signal and pileup_component in component_name:
                            instrument.pileup_components_in_signal[pileup_component] = component_name
                            if component_name not in to_remove:
                                to_remove.append(component_name)
            print('instrument.pileup_components_in_signal:',
                  instrument.pileup_components_in_signal)
        print('to_remove:', to_remove)
        all_components = [c for c in all_components if c not in to_remove]
        print('remaining components:', all_components)

        to_remove = []
        for instrument in self.instruments.values():
            instrument.background_components_in_signal = {}
            for component_name in all_components:
                if instrument.get_background_model() is not None:
                    for background_component in instrument.get_background_model().componentNames:
                        if background_component not in instrument.background_components_in_signal and background_component in component_name:
                            instrument.background_components_in_signal[background_component] = component_name
                            if component_name not in to_remove:
                                to_remove.append(component_name)
            print('instrument.background_components_in_signal:',
                  instrument.background_components_in_signal)
        print('to_remove:', to_remove)
        all_components = [c for c in all_components if c not in to_remove]

        print('remaining components:', all_components)

        if self.pileup_expression:
            self._set_pileup_links()
        if self.background_expression:
            self._set_background_links()

        # TODO: see if we can do this better.
        # Configure the factors to be correct.
        # This might just need to be something that doesn't get handled
        # in the class because it's very specific to the behavior I want.
        first_group = self.signal_groups[0]
        current_model = xspec.AllModels(first_group, self.current_model)
        factor_limits = '1 0.01 0.001 0.001 1000 1000'
        if 'constant' in current_model.componentNames:

            # Identify which constant to change, if there are multiple.
            for component_name in all_components:
                if 'constant' in component_name:
                    component = current_model.__dict__[component_name]
                    if not component.factor.link:
                        constant_to_change = component_name
                        break

            # Freeze the factor for spectrum 1.
            # print(self.current_model)
            # print(current_model)
            print(
                f'freezing signal factor for group {first_group} in model {self.current_model}')
            current_model.__dict__[constant_to_change].factor.link = ''
            current_model.__dict__[constant_to_change].factor.frozen = True

            # Free the offset factor between the groups (spectra).
            for group_num in self.signal_groups:
                if group_num != first_group:
                    current_model = xspec.AllModels(group_num, self.current_model).__dict__[
                        constant_to_change].factor = 1.0
                    current_model = xspec.AllModels(group_num, self.current_model).__dict__[
                        constant_to_change].factor.link = ''
                    current_model = xspec.AllModels(group_num, self.current_model).__dict__[
                        constant_to_change].factor.frozen = False
                    current_model = xspec.AllModels(group_num, self.current_model).__dict__[
                        constant_to_change].factor.values = factor_limits

        if len(self.background_groups) != 0:
            background_group = self.background_groups[0]
            print('background group:', background_group)
            if 'constant' in xspec.AllModels(background_group, 'background').componentNames:

                # Freeze the factor for first group.
                print(
                    f'freezing background factor for group {background_group}')
                xspec.AllModels(background_group,
                                'background').constant.factor.link = ''
                xspec.AllModels(background_group,
                                'background').constant.factor.frozen = True

                # Free the offset factor between the groups (spectra).
                for group_num in self.background_groups:
                    if group_num != background_group:
                        print(
                            f'freeing background factor for group {group_num}')
                        xspec.AllModels(
                            group_num, 'background').constant.factor = 1.0
                        xspec.AllModels(
                            group_num, 'background').constant.factor.link = ''
                        xspec.AllModels(
                            group_num, 'background').constant.factor.frozen = False

        return model

    # TODO: Implement this.

    def set_signal_parameter_limits():
        return

    def set_pileup_parameter_limits():
        return

    def _set_parameter_limits(
        self,
        config_file: str,
        tie_data_groups: bool
    ) -> dict[str, tuple]:
        '''# TODO: See if we can clean this up?
        TODO: also, need to configure limits per component,
        instead of applying the same limits to all components...
        '''

        self.config_file = config_file
        config = XSPECConfiguration(config_file)
        conf_dict = config.conf_dict
        applied = {}

        for instrument in self.instruments.values():
            group_num = instrument.signal_data_group
            model = xspec.AllModels(group_num, self.current_model)
            for component_name in model.componentNames:
                if 'expmodgauss' in component_name:
                    continue
                component = model.__dict__[component_name]
                for parameter_name in conf_dict:
                    if parameter_name in component.__dict__:
                        parameter = component.__dict__[parameter_name]
                        if group_num > 1 and tie_data_groups:  # TODO: Confirm if this is correct.
                            print(
                                f'group_num = {group_num}, and tie_data_groups = True, not setting limits for parameter {parameter_name}')
                            continue
                        limits = tuple(conf_dict[parameter_name].values())
                        parameter.values = np.array(limits).tolist()
                        applied[parameter_name] = conf_dict[parameter_name]
                        print(
                            f'setting parameter limits for group {group_num} component {component_name} parameter {parameter_name} ({parameter.index})\n{tuple(conf_dict[parameter_name].values())}\n')
                # Untie parameters if necessary.
                for parameter_name in component.parameterNames:
                    parameter = component.__dict__[parameter_name]
                    if not tie_data_groups:
                        parameter.link = ''

        return applied

    def set_gain(
        self,
        slopes: dict[str, float | tuple],
        offsets: dict[str, float | tuple],
        fit_slope: bool,
        fit_offset: bool,
        link_gains: bool = False
    ):
        '''slopes and offsets are dictionaries containing the
        instruments and their associated values.
        If a parameter is not fixed, it will be fitted.
        '''

        # TODO: Set gain errors.
        for instrument in self.instruments:
            group_num = self.instruments[instrument].signal_data_group
            response = xspec.AllData(group_num).response
            if instrument in slopes:
                response.gain.slope = slopes[instrument]
            else:
                print(
                    f'Instrument \"{instrument}\" not provided slopes. Skipping')
            if instrument in offsets:
                response.gain.offset = offsets[instrument]
                print(
                    f'Instrument \"{instrument}\" not provided offsets. Skipping')
            if not fit_slope:
                response.gain.slope.frozen = True
            if not fit_offset:
                response.gain.offset.frozen = True

        # Link the gain parameters.
        if link_gains:
            for link_num in self.signal_groups:
                if link_num != group_num:
                    response.gain.slope.link = xspec.AllData(
                        link_num).response.gain.slope
                    response.gain.offset.link = xspec.AllData(
                        link_num).response.gain.offset

    def fit(
        self,
        num_iterations: int = 50000,
        critical_delta: float = 0.01,
        fit_statistic: str = 'cstat',
        fit_error: bool = True
    ):
        '''
        fit_error will fit the errors on all the unfrozen parameters.
        '''

        xspec.Fit.nIterations = num_iterations
        xspec.Fit.criticalDelta = critical_delta
        xspec.Fit.statMethod = fit_statistic
        xspec.Fit.perform()

        if fit_error:
            error_str = ''
            for i, instrument in enumerate(self.instruments.values()):
                signal_model = instrument.get_signal_model(self.current_model)
                for component_name in signal_model.componentNames:
                    component = signal_model.__dict__[component_name]
                    for parameter_name in component.parameterNames:
                        parameter = component.__dict__[parameter_name]
                        if not parameter.frozen and '=' not in parameter.link:
                            corr_index = parameter.index
                            inst_index = (self.signal_groups).index(
                                instrument.signal_data_group)
                            corr_index += inst_index * signal_model.nParameters
                            error_str += f'{signal_model.name}:{corr_index} '

                # TODO: make this better
                models = []
                if instrument.get_pileup_model() is not None:
                    models.append(instrument.get_pileup_model())
                for model in models:
                    for component_name in model.componentNames:
                        component = model.__dict__[component_name]
                        for parameter_name in component.parameterNames:
                            parameter = component.__dict__[parameter_name]
                            if not parameter.frozen and '=' not in parameter.link:
                                corr_index = parameter.index
                                error_str += f'{model.name}:{corr_index} '

            print('error_str:', error_str)
            error_str = f'stop 5000,1, {error_str}'
            xspec.Fit.error(error_str)

        self._gather_results()
        xspec.Xset.closeLog()

    def _gather_results(self):

        results = {}
        for instrument in self.instruments.values():
            model_name = self.current_model
            data_group = instrument.signal_data_group
            results[instrument.name] = gather_model_data(
                model_name,
                data_group
            )
            if instrument.pileup_file is not None:
                model_name = instrument.pileup_model_name
                data_group = instrument.pileup_data_group
                results[f'{instrument.name} pileup'] = gather_model_data(
                    model_name,
                    data_group
                )

        self.results = results

        if CONFIG.debug:
            for name in self.results.keys():
                print(f'{name} result:')
                result = self.results[name]._asdict()
                for k, v in result.items():
                    print(f'\tKey: {k}')
                # if isinstance(self.results[k1], dict):
                #     for k2 in self.results[k1].keys():
                #         print('\t', k2)
                #         if isinstance(self.results[k1][k2], dict):
                #             for k3 in self.results[k1][k2].keys():
                #                 print('\t\t', k3)

    def get_parameter_string(self) -> str:
        '''TODO: Clean this up.
        Properly determine parameter uncertainties when the error command isn't called.
        TODO: This should probably be external to this class.
        '''

        full_string = ''
        full_component_names = xspec.AllModels(
            1, self.current_model).componentNames
        component_names = [c.split('_')[0] for c in full_component_names]
        unique = np.unique(component_names, return_counts=True)
        unique_components = {u: {'uses': 0, 'counts': c}
                             for u, c in zip(unique[0], unique[1])}

        # Add parameter values.
        for group_num in self.signal_groups:
            model = xspec.AllModels(group_num, self.current_model)

            for component_name in model.componentNames:
                comp_name = component_name.split('_')[0]
                postfix = ''
                if comp_name in unique_components:
                    counts = unique_components[comp_name]['counts']
                    if counts > 1:
                        unique_components[comp_name]['uses'] += 1
                        postfix = rf'$_{unique_components[comp_name]["uses"]}$'

                component = model.__dict__[component_name]
                mapping = MODEL_COMPONENT_MAPPING[comp_name]

                for parameter_name in component.parameterNames:
                    parameter = component.__dict__[parameter_name]
                    if not parameter.frozen and '=' not in parameter.link:

                        name, unit = mapping[parameter_name]
                        param = parameter.values[0] * unit
                        errors = np.array([*parameter.error[0:2]]) * unit
                        errors -= param

                        if comp_name in PARAMETER_CONVERSIONS and parameter_name in PARAMETER_CONVERSIONS[comp_name]:
                            param *= PARAMETER_CONVERSIONS[comp_name][parameter_name]
                            errors *= PARAMETER_CONVERSIONS[comp_name][parameter_name]

                        # Determine format.
                        if param.value >= 1000:
                            str_format = '{:0.{prec}E}'
                        else:
                            str_format = '{:0.{prec}f}'

                        # Determine prec.
                        mantissa = errors[1].value % 1
                        prec = 1
                        if mantissa != 0:
                            while (mantissa * 10 ** prec) < 1 and prec < 4:
                                prec += 1

                        param_str = str_format.format(param.value, prec=prec)
                        error_low = str_format.format(
                            errors[0].value, prec=prec)
                        error_high = str_format.format(
                            errors[1].value, prec=prec)
                        error_str = rf'$_{{{error_low}}}^{{+{error_high}}}$'

                        unit_str = f'[{param.unit}]'
                        if param.unit == u.Unit():
                            unit_str = ''

                        full_string += f'{name.upper()+postfix}: {param_str}{error_str} {unit_str}\n'

        # Add fit statistic(s).
        # for group_num in self.signal_groups:
        #     label = self.instruments[group_num-1]
        for instrument in self.instruments.values():
            spectrum = instrument.signal_spectrum
            bins = int(len(self.results[instrument.name].energy))
            full_string += f'{instrument.name} CSTAT: {spectrum.statistic:0.1f} ({bins} bins)\n'

        # Add gain.
        # TODO: Have this use the source_groups attribute.
        response = xspec.AllData(1).response
        if response.gain.isOn:
            full_string += f'GAIN SLOPE: {response.gain.slope.values[0]:0.3f}'
            if not response.gain.offset.frozen:
                full_string += f'\nGAIN OFFSET: {response.gain.offset.values[0]:0.3f}'

        return full_string
