from dallinger.nodes import Source
from dallinger.nodes import Agent
from dallinger.models import Info
from dallinger.models import Network, Node, Participant
from dallinger.recruiters import MTurkRecruiter # TODO: is the recruiter fixed?

from sqlalchemy import Integer, String, Float
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql.expression import cast

from operator import attrgetter
import random
import json
# import pysnooper

SINGLEPARENT = False

class ParticleFilter(Network):
    """Discrete fixed size generations with random transmission"""

    __mapper_args__ = {"polymorphic_identity": "particlefilter"}

    def __init__(self, generations, generation_size,replication):
        """Endow the network with some persistent properties."""
        self.property1 = repr(generations)
        self.property2 = repr(generation_size)
        self.max_size = generations * generation_size + 1 # add one to account for initial_source
        self.current_generation = 0
        self.replication = replication

    @property
    def generations(self):
        """The length of the network: the number of generations."""
        return int(self.property1)

    @property
    def generation_size(self):
        """The width of the network: the size of a single generation."""
        return int(self.property2)
    
    @hybrid_property
    def current_generation(self):
        """Make property3 current_generation."""
        return int(self.property3)

    @current_generation.setter
    def current_generation(self, current_generation):
        """Make current_generation settable."""
        self.property3 = repr(current_generation)

    @current_generation.expression
    def current_generation(self):
        """Make current_generation queryable."""
        return cast(self.property3, Integer)

    @hybrid_property
    def decision_index(self):
        """Make property4 decision_index."""
        return int(self.property4)

    @decision_index.setter
    def decision_index(self, decision_index):
        """Make decision_index settable."""
        self.property4 = repr(decision_index)

    @decision_index.expression
    def decision_index(self):
        """Make decision_index queryable."""
        return cast(self.property4, Integer)

    @hybrid_property
    def replication(self):
        """Make property3 replication."""
        return int(self.property5)

    @replication.setter
    def replication(self, replication):
        """Make replication settable."""
        self.property5 = repr(replication)

    @replication.expression
    def replication(self):
        """Make replication queryable."""
        return cast(self.property5, Integer)


    # @pysnooper.snoop()
    def add_node(self, node):
        
        node.generation = self.current_generation

        if self.current_generation == 0:
            parent = self._select_oldest_source()
            if parent is not None:
                parent.connect(whom=node)
                parent.transmit(to_whom=node)
        else:

            if SINGLEPARENT:
                sampled_parent = random.choice(list(filter(lambda node: int(node.generation) == int(self.current_generation) - 1, self.nodes(failed=False, type=Particle))))
                sampled_parent.connect(whom=node)
                sampled_parent.transmit(to_whom=node)
            else:
                parents = list(filter(lambda node: int(node.generation) == int(self.current_generation) - 1, self.nodes(failed=False, type=Particle)))
                for parent in parents:
                    parent.connect(whom=node)
                    parent.transmit(to_whom=node)

    def _select_oldest_source(self):
        return min(self.nodes(type=Source), key=attrgetter("creation_time"))


class Particle(Node):
    """The Rogers Agent."""

    __mapper_args__ = {"polymorphic_identity": "particle"}

    # @hybrid_property
    # def node_participant_id(self):
    #     """Convert property2 to participant."""
    #     return int(self.property2)

    # @participant_id.setter
    # def participant_id(self, participant_id):
    #     """Make participant_id settable."""
    #     self.property2 = repr(participant_id)

    # @participant_id.expression
    # def participant_id(self):
    #     """Make participant_id queryable."""
    #     return cast(self.property2, Integer)

    @hybrid_property
    def generation(self):
        """Convert property3 to genertion."""
        return int(self.property3)

    @generation.setter
    def generation(self, generation):
        """Make generation settable."""
        self.property3 = repr(generation)

    @generation.expression
    def generation(self):
        """Make generation queryable."""
        return cast(self.property3, Integer)

    @hybrid_property
    def participant_index(self):
        """Convert property3 to genertion."""
        return int(self.property4)

    @participant_index.setter
    def participant_index(self, participant_index):
        """Make participant_index settable."""
        self.property4 = repr(participant_index)

    @participant_index.expression
    def participant_index(self):
        """Make participant_index queryable."""
        return cast(self.property4, Integer)

    @hybrid_property
    def replication(self):
        """Make property5 replication."""
        return int(self.property5)

    @replication.setter
    def replication(self, replication):
        """Make replication settable."""
        self.property5 = repr(replication)

    @replication.expression
    def replication(self):
        """Make replication queryable."""
        return cast(self.property5, Integer)

    def __init__(self, contents=None, details = None, network = None, participant = None,replication=None,participant_index=None):
        super(Particle, self).__init__(network, participant)
        self.replication = replication
        self.participant_index = participant_index
        #self.participant_id = int(participant.id)


class WarOfTheGhostsSource(Source):
    """A Source that reads in a random story from a file and transmits it."""
    __mapper_args__ = {
        "polymorphic_identity": "war_of_the_ghosts_source"
    }

    def _contents(self):
        """Define the contents of new Infos.
        transmit() -> _what() -> create_information() -> _contents().
        """
        story = "superbugs.md"
        with open("static/stimuli/{}".format(story), "r") as f:
            return f.read()