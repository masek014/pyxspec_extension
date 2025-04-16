'''
Defines a simple interface to set XSPEC parameters since some parameters
(e.g. temperature) requires conversion to different units. 
'''

import astropy.units as u
import parse

from . import DEFAULT_PARAMETER_FILE

PARAM_CONF_FORMAT = '{param}({unit}):{init_value},{delta},{hard_min},{soft_min},{soft_max},{hard_max}\n'
KEV_TO_KELVIN = 11.6 * u.MK / u.keV
EARTH_NORM_TO_CM3 = 1 / (3.5557e-42 * (u.cm)**3)
XSPEC_PARAMETER_MAP = {
    'factor': 'factor',
    'temperature': 'kT',
    'emission_measure': 'norm',
    'break_energy': 'BreakE',
    'lower_photon_index': 'PhoIndx1',
    'upper_photon_index': 'PhoIndx2'
}


class XSPECConfiguration():
    '''Manages the parameter limits to give to XSPEC.
    Reads parameters set in a configuration file.
    '''

    def __init__(
        self,
        conf_file: str = None,
        norm_to_em_conversion: u.Quantity = EARTH_NORM_TO_CM3
    ):
        self.conf_file = conf_file or DEFAULT_PARAMETER_FILE
        self.norm_to_em_conversion = norm_to_em_conversion
        self.conf_dict = {}
        self._read_file()

    @property
    def file_name(self):
        '''Configuration file name.'''
        return self.conf_file.split('/')[-1]

    def _convert_to_xspec_units(self):
        '''Converts the "normal" units to the units used by XSPEC.'''
        for (param, values) in self.conf_dict.items():
            if param == 'kT':
                self.conf_dict[param] = {
                    k: (v / KEV_TO_KELVIN).value for (k, v) in values.items()}
            elif param == 'norm':
                self.conf_dict[param] = {
                    k: (v / EARTH_NORM_TO_CM3).value for (k, v) in values.items()}
            elif param == 'BreakE':
                self.conf_dict[param] = {
                    k: v.to(u.keV).value for (k, v) in values.items()}
            elif 'PhoIndx' in param:
                self.conf_dict[param] = {
                    k: v.value for (k, v) in values.items()}

    def _read_file(self):
        '''Load the parameter data from the configuration file.'''
        with open(self.conf_file, 'r', encoding='utf-8') as in_file:
            for line in in_file.readlines():
                if line[0] != '#' and line[0] != '\n':
                    line = line.replace(' ', '')
                    p = parse.parse(PARAM_CONF_FORMAT, line)
                    if p is not None:
                        values = p.named
                        param = values.pop('param').lower()
                        try:
                            xspec_param = XSPEC_PARAMETER_MAP[param]
                        except KeyError as e:
                            print(
                                f'Key \'{param}\' not defined in the XSPEC parameter map.\n{e}')
                        unit = values.pop('unit').replace('[]', '')
                        values = {
                            k: float(values[k]) * u.Unit(str(unit)) for k in values.keys()}
                        self.conf_dict[xspec_param] = {**values}
                    else:
                        line = line.replace('\n', '')
                        raise ValueError(
                            f'Provided configuration file \'{self.conf_file}\' contains an incorrectly formatted line: \'{line}\'\nCompare against the default configuration file found at {DEFAULT_PARAMETER_FILE} for proper formatting.')

        self._convert_to_xspec_units()
