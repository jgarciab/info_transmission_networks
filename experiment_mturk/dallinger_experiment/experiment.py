"""Bartlett's transmission chain experiment from Remembering (1932)."""

import logging
#import pysnooper
import ast
import random

from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from flask import Blueprint, Response

from sqlalchemy import and_, func
from sqlalchemy.sql.expression import cast
from dallinger.bots import BotBase
from dallinger.config import get_config
from dallinger.networks import Chain
from dallinger.experiment import Experiment
from dallinger import db

from datetime import datetime, time
from pytz import timezone

import re
import json

logger = logging.getLogger(__file__)


class Bartlett1932(Experiment):
    @property
    def public_properties(self):
        return {
        'generation_size': 3, # the width of the networks
        'read_multiple_versions': 0 # 0/1 for F/T of whether the initial generation should read multiple version of the same thing
        }

    """Define the structure of the experiment."""
    def __init__(self, session=None):
        """Call the same function in the super (see experiments.py in dallinger).

        A few properties are then overwritten.

        Finally, setup() is called.
        """
        super(Bartlett1932, self).__init__(session)
        from . import models  # Import at runtime to avoid SQLAlchemy warnings

        self.num_replications = 48
        self.models = models
        self.generation_size = self.public_properties['generation_size']
        self.initial_recruitment_size = 1 # self.num_replications*self.generation_size
        self.generations = 6
        self.num_experimental_networks_per_experiment = 1
        self.num_fixed_order_experimental_networks_per_experiment = 1
        #self.bonus_amount=1 # 1 for activating the extra bonus, 0 for deactivating it
        self.max_bonus_amount = 1
        self.read_multiple_versions = self.public_properties['read_multiple_versions']
        if session:
            self.setup()


    def configure(self):
        config = get_config()

    #@pysnooper.snoop(prefix = "@snoop SETUP: ")
    def setup(self):
        """Setup the networks"""

        """Create the networks if they don't already exist."""
        if not self.networks():
                
            # make an outer loop for number of replications
            #for replication_i in range(self.num_replications):
            #    for decision_index in range(self.num_fixed_order_experimental_networks_per_experiment,replication_i):
            #        network = self.create_network(role = 'experiment', decision_index = decision_index)
            #        print('NETWORK HERE:',network)
            #        self.models.WarOfTheGhostsSource(network=network)

            for replication_i in range(self.num_replications):
                for decision_i in range(self.num_fixed_order_experimental_networks_per_experiment):
                    network = self.create_network(role='experiment',decision_index=decision_i,replication=replication_i)
                    self.models.WarOfTheGhostsSource(network=network)

            self.session.commit()

    # Creating an instance of self.models.particle and returning
    # Instead of
    def create_node(self, network, participant):

        current_generation = self.get_current_generation()

        # Get all participant ids of those who are approved, working, submitted
        ok_participants = [p.id for p in self.models.Participant.query.filter(self.models.Participant.status.in_(["approved", "working", "submitted"]))]

        # Filter out current participant
        ok_participants.remove(participant.id)

        # Get everyone in the current replication and generation
        nodes_in_network = [node for node in self.models.Particle.query.filter(self.models.Particle.property3 == current_generation, self.models.Particle.property5 == str(network.replication))]

        # Only keep nodes where participants in relevant participant list
        def ok_node(node_obj):
            return node_obj.participant_id in ok_participants
        ok_nodes = list(filter(ok_node,nodes_in_network))

        # Extract property4 (participant index)
        used_participant_indices = [n.property4 for n in ok_nodes]
        used_participant_indices_ints = [eval(i) for i in used_participant_indices]

        # if (len(nodes_in_network) > 0):
        #     raise Exception('NODES ERROR', str(type(used_participant_indices[0]), str(type([]))))

        def filter_participants(participant_id):
            return not participant_id in used_participant_indices_ints

        all_participant_indices = list(range(0,self.generation_size))

        ok_participant_indices = list(filter(filter_participants,all_participant_indices))

        participant_index = int(ok_participant_indices[0])

        n = self.models.Particle(network=network,participant=participant, replication=network.replication,participant_index=participant_index)
        # You don't want to give two participants the same index ...
        # Get all nodes in this replication and generation and get their participant_index
        # you can call network.nodes OR:
        #print(network.nodes)
        #index = self.models.Network.query.filter(self.models.Network.property4 == repr(completed_decisions)).filter_by(full = False).one()
        #something like self.models.Network.query.filter(self.models.Network.property4 == repr(completed_decisions)).filter_by(full = False).one()
        #n.participant_index = network.id
        # self.session.commit() or something like this
        return n

    def create_network(self, role, decision_index,replication):
        """Return a new network."""
        # add argument for replication to particle filter in models.py
        net = self.models.ParticleFilter(generations = self.generations, generation_size = self.generation_size,replication=replication)
        net.role = role
        net.decision_index = decision_index
        #net.replication = replication
        self.session.add(net)
        return net

    def add_node_to_network(self, node, network):
        """Add node to the chain and receive transmissions."""
        network.add_node(node)
        parents = node.neighbors(direction="from")
        #if len(parents):
        #    parent = parents[0]
        #    parent.transmit()
        node.receive()

    def get_submitted_text(self, participant):
        """The text a given participant submitted"""
        # import rpdb; rpdb.set_trace()
        # submitted_dict = ast.literal_eval(submitted_text)
        submitted_text = participant.nodes()[0].infos()[0].contents
        #print('-- -- -- -- SUBMITTED TEXT -- -- -- --')
        #print(submitted_text)
        if submitted_text[0]=='{':
            submitted_dict = ast.literal_eval(submitted_text)
            submitted_text = submitted_dict['response']
        return submitted_text

    #@pysnooper.snoop()
    def get_read_text(self, participant):
        """The text that a given participant was shown to memorize"""
        # node = participant.nodes()[0]
        # node_length = len(node.all_incoming_vectors)
        # contents_list = []
        # for indexi in range(node_length):
        #     curr_incoming = node.all_incoming_vectors[indexi]
        #     curr_origin = curr_incoming.origin
        #     curr_text = curr_origin.infos()[0].contents
        #     if curr_text[0]=='{':
        #         curr_dict = ast.literal_eval(curr_text)
        #         curr_text = curr_dict['response']
        #     contents_list.append(curr_text)
        # return contents_list
        participant_dict = ast.literal_eval(participant.nodes()[0].infos()[0].contents)
        return participant_dict['read_stories']

    ### -- -- -- -- -- -- ###
    ### -- -- -- -- -- -- ###
    ### NEW ATTENTION CHECK ###
    ### -- -- -- -- -- -- ###
    ### -- -- -- -- -- -- ###
    def tokenize_sentence(self, text):
        """Return a list of words from a string"""
        return re.findall(r'\w+', text)

    def calculate_jaccard_sim(self,input_list,response):
        """Return the Jaccard similarity between two strings at the word level removing stopwords"""
        stopwords = {"couldn't", "you're", 'same', 'she', 'so', 'hers', 'being', "you'll", 'more', 'those', 'm', "shan't", 've', 'hadn', 'up', 'her', "wasn't", 'herself', 'are', 'at', 'while', 'can', 'few', 'them', 'yourselves', 'its', 'i', 'we', 'yourself', 'nor', 'couldn', 'it', 'were', 'll', 'ain', "mightn't", "wouldn't", 'the', 'after', 'ma', "hadn't", 're', 'is', 'had', 'until', "didn't", 'themselves', 'd', "haven't", 'they', 'their', "won't", 's', 'didn', "needn't", 'through', 'or', 'needn', 'won', 'that', 'who', 'over', 'some', 'not', 'against', 'down', 'above', 'your', 'how', 'isn', 'once', 'ourselves', 'a', 'to', 'between', 'his', 'shouldn', 'why', 'hasn', 'doing', 'no', 'before', 'now', 'very', 'wasn', 'he', "you'd", "don't", 'further', 'itself', 'other', 'whom', 'these', 'mightn', 'o', 'do', 'when', 'because', 'where', 'into', 'each', 'on', 't', "she's", "it's", "weren't", 'did', 'just', 'out', 'which', 'wouldn', 'this', 'shan', 'below', 'too', "should've", 'mustn', 'during', 'doesn', 'both', "doesn't", 'as', 'my', 'him', 'weren', "that'll", 'what', 'ours', 'himself', 'under', "aren't", 'theirs', 'for', 'most', "shouldn't", 'than', 'y', 'yours', 'will', 'any', 'me', 'here', 'off', 'if', 'has', 'with', 'and', 'does', 'again', "mustn't", 'in', 'be', 'having', 'about', 'was', 'only', 'such', 'don', 'am', 'have', 'own', 'of', "you've", 'from', 'by', 'all', 'but', 'our', 'should', 'haven', "hasn't", 'an', "isn't", 'there', 'been', 'aren', 'then', 'you', 'myself'}
        
        input_list = " ".join(input_list)

        # Tokenize word
        input_list = self.tokenize_sentence(input_list.lower())
        response = self.tokenize_sentence(response.lower())
        set1 = set(input_list) - stopwords
        set2 = set(response) - stopwords
        return len(set1.intersection(set2))/len(set1.union(set2))


    def levenstein_ratio(self,string_1, string_2):
        """
        Calculates the Levenshtein distance between two strings.
        This version uses an iterative version of the Wagner-Fischer algorithm.
        """
        if string_1 == string_2:
            return 1
            #return 0
        len_1 = len(string_1)
        len_2 = len(string_2)
        if len_1 == 0:
            return 0
            #return len_2
        if len_2 == 0:
            return 0
            #return len_1
        if len_1 > len_2:
            string_2, string_1 = string_1, string_2
            len_2, len_1 = len_1, len_2
        d0 = [i for i in range(len_2 + 1)]
        d1 = [j for j in range(len_2 + 1)]
        for i in range(len_1):
            d1[0] = i + 1
            for j in range(len_2):
                cost = d0[j]
                if string_1[i] != string_2[j]:
                    # substitution
                    cost += 2
                    # insertion
                    x_cost = d1[j] + 1
                    if x_cost < cost:
                        cost = x_cost
                    # deletion
                    y_cost = d0[j + 1] + 1
                    if y_cost < cost:
                        cost = y_cost
                d1[j + 1] = cost
            d0, d1 = d1, d0
        lensum = len(string_1)+len(string_2)
        return (lensum-d0[-1])/lensum

    def calculate_lev_sim(self,input, response):
        """Return a measure of the similarity between two texts"""
        #return ratio(input.lower(),response.lower())
        return self.levenstein_ratio(input.lower(), response.lower())

    def calculate_sentence_sim(self,input_list, response):
        """"Return the similarity at the between the list of inputs and the output"""    
        input_list  = ". ".join(input_list)
        input = [_ for _ in re.split("\. ?|\n", input_list)]

        response_len = len(response)
        response = [_ for _ in re.split("\. ?|\n", response)]

        # Levensthein similarity with each sentence in output
        dist = 0
        for sentence in response:
            sims = [self.calculate_lev_sim(sentence, _) for _ in input] # Levensthein similarity
            dist += max(sims)*len(sentence)
        # print('CUSTOM LEV')
        # print(self.levenstein_distance(input.lower(),response.lower()))
        return dist/response_len

    #def accept(input_list, response):
    def attention_check(self,participant):
        # Get response info
        read_text = self.get_read_text(participant)
        response = self.get_submitted_text(participant)
        current_generation = self.get_current_generation()
        
        # Participant didn't write anything
        if len(response) == 0:
            return False

        # Participant wrote too much and it might crash
        if len(response) > 3000:
            return False
        
        len_input = sum([len(_) for _ in read_text])/len(read_text)
        jac_sim = self.calculate_jaccard_sim(read_text, response)
        sentence_sim = self.calculate_sentence_sim(read_text,response)
        # Small input, avoid nonsensical
        if len_input < 150:
            if (sentence_sim < 0.4):
                return False
        else:
            # Every generation avoid C&P sentences or nonsensical
            if (sentence_sim < 0.4) | \
                (sentence_sim > 0.7) | \
                (jac_sim < 0.03):
                return False
            # First generation only: avoid paraphrasing or nonsensical
            elif current_generation==0:
                lev_sim = self.calculate_lev_sim(read_text[0], response)
                if (lev_sim > 0.4) | \
                    (lev_sim < 0.1):
                    return False
        return True

    #@pysnooper.snoop()
    def bonus(self, participant):
        """The bonus to be awarded to the given participant.
        Return the value of the bonus to be paid to `participant`.
        """
        response = self.get_submitted_text(participant)
        if (len(response)==0):
            return 0
        passed_check = self.attention_check(participant)
        if (passed_check):
            return self.max_bonus_amount
        return 0

        # text_input=self.get_read_text(participant)
        # total_performance = 0
        # len_text = 0
        # for storyi in range(len(text_input)):
        #     len_text += len(str(text_input[storyi]).split(' '))
        #     curr_performance = self.text_similarity(
        #     self.get_submitted_text(participant),
        #     text_input[storyi])
        #     total_performance += curr_performance
        # average_performance = total_performance/len(text_input)
        # if participant.nodes()[0].generation == 0 and self.read_multiple_versions==1:
        #     text_reward = (0.002 * len_text)*3 # read multiple versions of the same thing
        # else:
        #     text_reward = (0.002 * len_text)
        # payout = round(text_reward, 2)
        # if average_performance <= 0.02:
        #     payout = payout / 4
        # return min(payout, self.max_bonus_amount)

    # #@pysnooper.snoop()
    def get_network_for_existing_participant(self, participant, participant_nodes):
        """Obtain a netwokr for a participant who has already been assigned to a condition by completeing earlier rounds"""

        # which networks has this participant already completed?
        networks_participated_in = [node.network_id for node in participant_nodes]
        
        # How many decisions has the particiapnt already made?
        completed_decisions = len(networks_participated_in)

        # When the participant has completed all networks in their condition, their experiment is over
        # returning None throws an error to the fronted which directs to questionnaire and completion
        if completed_decisions == self.num_experimental_networks_per_experiment:
            return None

        nfixed = self.num_fixed_order_experimental_networks_per_experiment

        # If the participant must still follow the fixed network order
        if completed_decisions < nfixed:
            # find the network that is next in the participant's schedule
            # match on completed decsions b/c decision_index counts from zero but completed_decisions count from one
            return self.models.Network.query.filter(self.models.Network.property4 == repr(completed_decisions)).filter_by(full = False).one()

        # If it is time to sample a network at random
        else:
            # find networks which match the participant's condition and werent' fixed order nets
            matched_condition_experimental_networks = self.models.Network.query.filter(cast(self.models.Network.property4, Integer) >= nfixed).filter_by(full = False).all()
            
            # subset further to networks not already participated in (because here decision index doesnt guide use)
            availible_options = [net for net in matched_condition_experimental_networks if net.id not in networks_participated_in]
            
            # choose randomly among this set
            chosen_network = random.choice(availible_options)

        return chosen_network

    #@pysnooper.snoop(prefix = "@snoop: ")
    def get_network_for_new_participant(self, participant):
        #key = "experiment.py >> get_network_for_new_participant ({}); ".format(participant.id)

        available_networks = self.models.ParticleFilter.query.filter(self.models.ParticleFilter.property4 == repr(0)).all()
        sorted_networks = sorted(available_networks, key=lambda d: d.id) 

        # print(self.models.ParticleFilter.query.filter(self.models.ParticleFilter.property4 == repr(0)).all())

        #first_network = available_networks[0].property5
        #second_network = available_networks[1].property5

        #occupancy_counts = (
        #    db.session.query(
        #    func.count(self.models.Particle.participant_id.distinct()).label('count'), self.models.ParticleFilter.property5)
        #    .join(self.models.ParticleFilter)
        #    .join(self.models.Participant)
        #    .group_by(self.models.ParticleFilter.property5)  
        #    .filter_by(failed = False)
        #    .filter(self.models.ParticleFilter.property4 == repr(0))
        #     .filter(self.models.Participant.status == "approved")
        #).all()

        #second_thing = (
        #    db.session.query(
        #        func.count(self.models.Particle.participant_id.distinct()).label('count'), self.models.ParticleFilter.property5, self.models.Participant.status)
        #        .join(self.models.ParticleFilter)
        #        .group_by(self.models.ParticleFilter.property5, self.models.Participant.status)  
        #        .filter_by(failed = False)
        #        .filter(self.models.ParticleFilter.property4 == repr(0))
        #).all()

        #third_thing =   (
        #    db.session.query(
        #    func.count(self.models.Particle.participant_id.distinct()).label('count'), self.models.ParticleFilter.property5, self.models.Participant.status)
        #    .join(self.models.ParticleFilter)
        #    .join(self.models.Participant)
        #    .group_by(self.models.ParticleFilter.property5, self.models.Participant.status)  
        #    .filter_by(failed = False)
        #    .filter(self.models.ParticleFilter.property4 == repr(0))
        #).all()

        occupancy_counts = (
            db.session.query(
            func.count(self.models.Particle.participant_id.distinct()).label('count'), self.models.ParticleFilter.property5)
            .join(self.models.ParticleFilter)
            .join(self.models.Participant)
            .group_by(self.models.ParticleFilter.property5)  
            .filter_by(failed = False)
            .filter(self.models.ParticleFilter.property4 == repr(0))
            .filter(self.models.Participant.status.in_(["approved", "working", "submitted"]))
        ).all()

        # print('-- -- -- -- OCCUPANCY COUNTS -- -- -- -- ')
        # print(occupancy_counts)

        if len(occupancy_counts)>=1:
            if len(occupancy_counts)==self.num_replications:
                min_val = 10000 # whole thing is hacky, but can make cleaner later
                min_val_list = []
                for replication_i in range(len(occupancy_counts)):
                    curr_val = occupancy_counts[replication_i][0]
                    if curr_val<min_val:
                        min_val_list = [int(occupancy_counts[replication_i][1])] # append with the replication int
                        min_val=curr_val
                    elif curr_val==min_val:
                        min_val_list.append(int(occupancy_counts[replication_i][1]))
            elif len(occupancy_counts) < self.num_replications: # in initial generation here,
                curr_generation_list = list(range(0,self.num_replications))
                for replication_i in range(len(occupancy_counts)):
                    curr_replication = int(occupancy_counts[replication_i][1])
                    curr_generation_list.remove(curr_replication)
                min_val_list = curr_generation_list
        else:
            min_val_list = list(range(0,self.num_replications))

        current_replication = random.choice(min_val_list)

        #random_choice_list = []
        #for replication_i in range(len(occupancy_counts)):
        #    curr_in_replication = occupancy_counts[replication_i][0]
        #    curr_remaining = num_allowed-curr_in_replication
        #    curr_list = [replication_i]*curr_remaining
        #    random_choice_list = random_choice_list+curr_list

        #print('OCCUPANCY COUNTS:',occupancy_counts)
        #print("RANDOM CHOICE LIST",random_choice_list)
        #print("RANDOM CHOICE",random.choice(random_choice_list))
        # nothing to stop this giving them both replication 0

        # try:
        #     test = available_networks[current_replication]
        # except:
        #     raise Exception('Indexing error' + '---' + str(available_networks) + '---' + str(occupancy_counts) + '---' +str(current_replication))
        return sorted_networks[current_replication]

    #@pysnooper.snoop()
    def get_network_for_participant(self, participant):
        """Find a network for a participant."""
        key = "experiment.py >> get_network_for_participant ({}); ".format(participant.id)
        participant_nodes = participant.nodes()
        # print('PARTICIPANT NODES',participant_nodes)
        if not participant_nodes:
            chosen_network = self.get_network_for_new_participant(participant)
        else:
            chosen_network = self.get_network_for_existing_participant(participant, participant_nodes)

        if chosen_network is not None:
            self.log("Assigned to network: {}; Decsion Index: {};".format(chosen_network.id, chosen_network.decision_index), key)

        else:
            self.log("Requested a network but was assigned None.".format(len(participant_nodes)), key)

        return chosen_network

    # @pysnooper.snoop()
    def get_current_generation(self):
        network = self.models.ParticleFilter.query.first()
        return repr(int(network.property3))

    def rollover_generation(self):
        for network in self.models.ParticleFilter.query.all():
            network.current_generation = int(network.current_generation) + 1
        self.log("Rolled over all network to generation {}".format(network.current_generation), "experiment.py >> rollover_generation: ")

    # @pysnooper.snoop()
    def recruit(self):
        """Recruit one participant at a time until all networks are full."""
        if self.networks(full=False):
            current_generation = self.get_current_generation()

            completed_participant_ids = [p.id for p in self.models.Participant.query.filter_by(failed = False, status = "approved")]
            
            # particle.property3 = generation
            completed_nodes_this_generation = self.models.Particle.query.filter(
                                                                            self.models.Particle.property3 == current_generation, \
                                                                            self.models.Particle.participant_id.in_(completed_participant_ids)) \
                                                                        .count() 

            if completed_nodes_this_generation == self.generation_size*self.num_replications:
              self.rollover_generation()
              if self.is_valid_time():
                  self.recruiter.recruit(n=self.generation_size*self.num_replications)

        else:
            self.recruiter.close_recruitment()


    def is_valid_time(self):
        return False
        curfew_start = time(19, 0)  # UTC, 2:00pm NYC time
        curfew_end = time(11, 0)  # UTC, (11,0) == 7am NYC time
        UTC = timezone("UTC")
        now = datetime.now(UTC)
        self.log("Time now (UTC) is: {}".format(now.time()))
        if curfew_start < curfew_end:
            in_curfew_window = (now.time() >= curfew_start) and (
                now.time() <= curfew_end
            )
        else:
            in_curfew_window = (now.time() >= curfew_start) or (
                now.time() <= curfew_end
            )
        return False if in_curfew_window else True


extra_routes = Blueprint(
    "extra_routes",
    __name__,
    template_folder="templates",
    static_folder="static",
)

@extra_routes.route(
    "/recruitbutton/<int:nparticipants>/", methods=["GET"]
)

def recruitbutton(nparticipants):
    try:
        from . import models

        exp = Bartlett1932(db.session)

        exp.recruiter.recruit(n=nparticipants)
        exp.log(
            "Made {} additional recruitments.".format(nparticipants),
            "experiment.py >> /recruitbutton",
        )
        return Response(
            json.dumps({"status": "Success!"}),
            status=200,
            mimetype="application/json",
        )

    except Exception:
        db.logger.exception("Error fetching node info")
        return Response(status=403, mimetype="application/json")

@extra_routes.route(
    "/kickOutParticipant/<int:participantId>/", methods=["GET"]
)

def kickOutParticipant(participantId):
    try:
        from . import models
        exp = Bartlett1932(db.session)
        kicked_out = False
        participant = exp.models.Participant.query.filter_by(id=participantId).first()
        if participant.mode not in ['live','debug','sandbox']:
            participant.fail()
            participant.status = 'returned'
            db.session.commit()
            kicked_out = True
            # Recruit a new participant
            exp.recruiter.recruit(n=1)
        return Response(
            json.dumps({"status": "Success!","kicked_out":kicked_out}),
            status=200,
            mimetype="application/json",
        )

    except Exception:
        db.logger.exception("Error fetching node info")
        return Response(status=403, mimetype="application/json")
