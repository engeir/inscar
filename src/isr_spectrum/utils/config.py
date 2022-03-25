from typing import Tuple, Union
import numpy as np
import scipy.constants as const
import attr


def is_odd(_, attribute, value):
    if value % 2 == 0:
        raise ValueError(f"{attribute} must be odd")


def is_positive(_, attribute, value):
    if not isinstance(value, (int, float)):
        raise ValueError(f"{attribute} must be a positive number")
    if value <= 0:
        raise ValueError(f"{attribute} must be positive")


def is_nonnegative(_, attribute, value):
    if not isinstance(value, (int, float)):
        raise ValueError(f"{attribute} must be a non-negative number")
    if value < 0:
        raise ValueError(f"{attribute} must be non-negative")


def is_range_tuple(_, attribute, value):
    if not isinstance(value, tuple):
        raise ValueError(f"{attribute} must be a tuple")
    if len(value) != 2:
        raise ValueError(f"{attribute} must be a tuple of length 2")
    if isinstance(value[0], (int, float)) and isinstance(value[1], (int, float)):
        if value[0] >= value[1]:
            raise ValueError(f"{attribute} must be a tuple of increasing values")
    else:
        raise ValueError(f"{attribute} must be a tuple of int/floats")


@attr.s(auto_attribs=True)
class Particle:
    gordeyev_upper_lim: Union[float, int] = attr.ib(
        default=1.5e-4,
        validator=is_positive,
        on_setattr=attr.setters.validate,
    )
    gordeyev_size: int = attr.ib(
        default=8e4 + 1,
        validator=is_odd,
        on_setattr=[attr.setters.convert, attr.setters.validate],
        converter=int,
    )
    gordeyev_exp: int = attr.ib(
        default=3,
        validator=is_odd,
        on_setattr=[attr.setters.convert, attr.setters.validate],
        converter=int,
    )
    velocity_upper_lim: Union[float, int] = attr.ib(
        default=6e6,
        validator=is_positive,
        on_setattr=attr.setters.validate,
    )
    velocity_size: int = attr.ib(
        default=4e4 + 1,
        validator=is_odd,
        on_setattr=[attr.setters.convert, attr.setters.validate],
        converter=int,
    )
    velocity_exp: int = attr.ib(
        default=3,
        validator=is_odd,
        on_setattr=[attr.setters.convert, attr.setters.validate],
        converter=int,
    )
    temperature: float = attr.ib(
        default=5000,
        validator=is_positive,
        on_setattr=attr.setters.validate,
    )
    collision_frequency: float = attr.ib(
        default=0,
        validator=is_nonnegative,
        on_setattr=attr.setters.validate,
    )
    mass: float = attr.ib(
        default=const.m_e,
        validator=is_positive,
        on_setattr=attr.setters.validate,
    )
    number_density: float = attr.ib(
        default=2e11,
        validator=is_positive,
        on_setattr=attr.setters.validate,
    )
    kappa: float = attr.ib(
        default=20,
        validator=is_positive,
        on_setattr=attr.setters.validate,
    )

    @property
    def velocity_axis(self) -> np.ndarray:
        return (
            np.linspace(
                0,
                self.velocity_upper_lim ** (1 / self.velocity_exp),
                self.velocity_size,
            )
        ) ** self.velocity_exp

    @property
    def gordeyev_axis(self) -> np.ndarray:
        return (
            np.linspace(
                0,
                self.gordeyev_upper_lim ** (1 / self.gordeyev_exp),
                self.gordeyev_size, dtype=np.double,
            )
        ) ** self.gordeyev_exp


@attr.s(auto_attribs=True)
class Parameters:
    radar_frequency: float = attr.ib(
        default=430e6,
        validator=is_positive,
        on_setattr=attr.setters.validate,
    )
    frequency_range: Tuple = attr.ib(
        default=(-2e6, 2e6),
        validator=is_range_tuple,
        on_setattr=attr.setters.validate,
    )
    frequency_size: int = attr.ib(
        default=1e4 + 1,
        validator=is_odd,
        on_setattr=[attr.setters.convert, attr.setters.validate],
        converter=int,
    )
    frequency_exp: int = attr.ib(
        default=1,
        validator=is_odd,
        on_setattr=[attr.setters.convert, attr.setters.validate],
        converter=int,
    )
    magnetic_field_strength: float = attr.ib(
        default=35000e-9,
        validator=is_positive,
        on_setattr=attr.setters.validate,
    )
    aspect_angle: Union[int, float] = attr.ib(
        default=45,
        validator=attr.validators.instance_of((int, float)),
        on_setattr=[attr.setters.convert, attr.setters.validate],
        converter=np.radians,
    )

    @property
    def linear_frequency(self) -> np.ndarray:
        return (
            np.linspace(
                self.frequency_range[0], self.frequency_range[1], self.frequency_size
            )
            / self.frequency_range[1]
        ) ** self.frequency_exp * self.frequency_range[1]

    @property
    def angular_frequency(self) -> np.ndarray:
        return self.linear_frequency * 2 * np.pi

    @property
    def radar_wavenumber(self) -> float:
        return -2 * self.radar_frequency * 2 * np.pi / const.c
