import math

from autofit.mapper.prior_model.prior_model import PriorModel
from autofit.mapper.prior_model.util import PriorModelNameValue
from autofit.mapper.prior.prior import Prior
from autofit.mapper.prior_model.abstract import AbstractPriorModel

# noinspection PyAbstractClass
from autofit.mapper.prior_model import attribute_pair


class Circle:
    def __init__(self, radius):
        self.radius = radius

    def with_circumference(self, circumference):
        self.circumference = circumference

    @property
    def circumference(self):
        return self.radius * 2 * math.pi

    @circumference.setter
    def circumference(self, circumference):
        self.radius = circumference / (2 * math.pi)


class GeometryProfile:
    def __init__(self, centre=(0.0, 0.0)):
        """Abstract GeometryProfile, describing an object with y, x cartesian
        coordinates """
        self.centre = centre

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class SphProfile(GeometryProfile):
    def __init__(self, centre=(0.0, 0.0)):
        """ Generic circular profiles class to contain functions shared by light and
        mass profiles.

        Parameters
        ----------
        centre
            The (y,x) coordinates of the origin of the profile.
        """
        super(SphProfile, self).__init__(centre)


class EllProfile(SphProfile):
    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, angle=0.0):
        """ Generic elliptical profiles class to contain functions shared by light
        and mass profiles.

        Parameters
        ----------
        centre
            The (y,x) coordinates of the origin of the profiles
        axis_ratio
            Ratio of profiles ellipse's minor and major axes (b/a)
        angle : float
            Rotational angle of profiles ellipse counter-clockwise from positive x-axis
        """
        super(EllProfile, self).__init__(centre)
        self.axis_ratio = axis_ratio
        self.angle = angle


class MassProfile:
    def surface_density_func(self, eta):
        raise NotImplementedError("surface_density_at_radius should be overridden")

    def surface_density_from_grid(self, grid):
        pass

    def potential_2d_from_grid(self, grid):
        pass

    def deflections_2d_from_grid(self, grid):
        raise NotImplementedError("deflections_2d_from_grid should be overridden")


# noinspection PyAbstractClass
class EllMassProfile(EllProfile, MassProfile):
    def __init__(self, centre=(0.0, 0.0), axis_ratio=1.0, angle=0.0):
        """
        Abstract class for elliptical mass profiles.

        Parameters
        ----------
        centre
            The origin of the profile
        axis_ratio
            Ellipse's minor-to-major axis ratio (b/a)
        angle : float
            Rotation angle of profile's ellipse counter-clockwise from positive x-axis
        """
        super(EllMassProfile, self).__init__(centre=centre, axis_ratio=axis_ratio, angle=angle)
        self.axis_ratio = axis_ratio
        self.angle = angle


# noinspection PyAbstractClass
class EllIsothermalCored(EllProfile):
    def __init__(
            self,
            centre=(0.0, 0.0),
            axis_ratio=1.0,
            angle=0.0,
            einstein_radius=1.0,
            core_radius=0.05,
    ):
        """
        Represents a cored elliptical isothermal density distribution, which is
        equivalent to the elliptical power-law
        density distribution for the value slope=2.0

        Parameters
        ----------
        centre
            The image_grid of the origin of the profiles
        axis_ratio
            Ell mass profile's minor-to-major axis ratio (b/a)
        angle : float
            Rotation angle of mass profile's ellipse counter-clockwise from positive
            x-axis
        einstein_radius : float
            Einstein radius of power-law mass profiles
        core_radius : float
            The radius of the inner core
        """

        super(EllIsothermalCored, self).__init__(
            centre=centre, axis_ratio=axis_ratio, angle=angle,
        )
        self.einstein_radius = einstein_radius
        self.core_radius = core_radius


class EllSersic(EllProfile):
    def __init__(
            self,
            centre=(0.0, 0.0),
            axis_ratio=1.0,
            angle=0.0,
            normalization=0.1,
            effective_radius=0.6,
            sersic_index=4.0,
    ):
        """ The elliptical Sersic profile, used for fitting a model_galaxy's light.

        Parameters
        ----------
        centre
            The (y,x) origin of the light profile.
        axis_ratio
            Ratio of light profiles ellipse's minor and major axes (b/a).
        angle : float
            Rotation angle of light profile counter-clockwise from positive x-axis.
        normalization
            Overall normalization normalisation of the light profiles (electrons per
            second).
        effective_radius
            The circular radius containing half the light of this profile.
        sersic_index : Int
            Controls the concentration of the of the light profile.
        """
        super().__init__(
            centre=centre, axis_ratio=axis_ratio, angle=angle,
        )
        self.normalization = normalization
        self.effective_radius = effective_radius
        self.sersic_index = sersic_index


class EllSersicCore(EllSersic):
    def __init__(
            self,
            centre=(0.0, 0.0),
            axis_ratio=1.0,
            angle=0.0,
            normalization=0.1,
            effective_radius=0.6,
            sersic_index=4.0,
            radius_break=0.01,
            normalization_break=0.05,
            gamma=0.25,
            alpha=3.0,
    ):
        """ The elliptical cored-Sersic profile, used for fitting a model_galaxy's
        light.

        Parameters
        ----------
        centre
            The (y,x) origin of the light profile.
        axis_ratio
            Ratio of light profiles ellipse's minor and major axes (b/a).
        angle : float
            Rotation angle of light profile counter-clockwise from positive x-axis.
        normalization
            Overall normalization normalisation of the light profiles (electrons per
            second).
        effective_radius
            The circular radius containing half the light of this profile.
        sersic_index : Int
            Controls the concetration of the of the light profile.
        radius_break : Float
            The break radius separating the inner power-law (with logarithmic slope
            gamma) and outer Sersic function.
        normalization_break : Float
            The normalization at the break radius.
        gamma : Float
            The logarithmic power-law slope of the inner core profiles
        alpha :
            Controls the sharpness of the transition between the inner core / outer
            Sersic profiles.
        """
        super(EllSersicCore, self).__init__(
            centre=centre,
            axis_ratio=axis_ratio,
            angle=angle,
            normalization=normalization,
            effective_radius=effective_radius,
            sersic_index=sersic_index
        )
        self.radius_break = radius_break
        self.normalization_break = normalization_break
        self.alpha = alpha
        self.gamma = gamma


class EllExponential(EllSersic):
    def __init__(
            self,
            centre=(0.0, 0.0),
            axis_ratio=1.0,
            angle=0.0,
            normalization=0.1,
            effective_radius=0.6,
    ):
        """ The elliptical exponential profile, used for fitting a model_galaxy's light.

        This is a subset of the elliptical Sersic profile, specific to the case that
        sersic_index = 1.0.

        Parameters
        ----------
        centre
            The (y,x) origin of the light profile.
        axis_ratio
            Ratio of light profiles ellipse's minor and major axes (b/a).
        angle : float
            Rotation angle of light profile counter-clockwise from positive x-axis.
        normalization
            Overall normalization normalisation of the light profiles (electrons per
            second).
        effective_radius
            The circular radius containing half the light of this profile.
        """
        super(EllExponential, self).__init__(
            centre=centre,
            axis_ratio=axis_ratio,
            angle=angle,
            normalization=normalization,
            effective_radius=effective_radius,
            sersic_index=1.0
        )



class EllGaussian(EllProfile):

    def __init__(
            self, centre=(0.0, 0.0), axis_ratio=1.0, angle=0.0, normalization=0.1, sigma=0.01
    ):
        """ The elliptical Gaussian profile.

        Parameters
        ----------
        centre
            The (y,x) origin of the light profile.
        axis_ratio
            Ratio of light profiles ellipse's minor and major axes (b/a).
        angle : float
            Rotation angle of light profile counter-clockwise from positive x-axis.
        normalization
            Overall normalization normalisation of the light profiles (electrons per
            second).
        sigma : float
            The full-width half-maximum of the Gaussian.
        """
        super(EllGaussian, self).__init__(centre, axis_ratio, angle)

        self.normalization = normalization
        self.sigma = sigma


class Redshift:
    def __init__(self, redshift):
        self.redshift = redshift


class Galaxy:
    def __init__(
            self,
            light_profiles: list = None,
            mass_profiles: list = None,
            redshift=None,
            **kwargs
    ):
        self.redshift = redshift
        self.light_profiles = light_profiles
        self.mass_profiles = mass_profiles
        self.kwargs = kwargs


class Tracer:
    def __init__(self, lens_galaxy: Galaxy, source_galaxy: Galaxy, grid):
        self.lens_galaxy = lens_galaxy
        self.source_galaxy = source_galaxy
        self.grid = grid


# # noinspection PyAbstractClass
# class GalaxyModel(af.AbstractPriorModel):
#     def instance_for_arguments(self, arguments):
#         try:
#             return Galaxy(redshift=self.redshift.instance_for_arguments(arguments))
#         except AttributeError:
#             return Galaxy()


# noinspection PyAbstractClass
class GalaxyModel(PriorModel):
    def __init__(self, model_redshift=False, **kwargs):
        super().__init__(
            Galaxy,
            redshift=PriorModel(Redshift) if model_redshift else None,
            **kwargs
        )

    @property
    def instance_tuples(self):
        return []

    @property
    @attribute_pair.cast_collection(
        attribute_pair.PriorNameValue
    )
    def unique_prior_tuples(self):
        return (
            [item for item in self.__dict__.items() if isinstance(item[1], Prior)]
            + [("redshift", self.redshift.redshift)]
            if self.redshift is not None
            else []
        )

    @property
    @attribute_pair.cast_collection(PriorModelNameValue)
    def flat_prior_model_tuples(self):
        return [
            item
            for item in self.__dict__.items()
            if isinstance(item[1], AbstractPriorModel)
        ]
