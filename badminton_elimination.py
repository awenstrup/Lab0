'''Code file for badminton elimination lab created for Advanced Algorithms
Spring 2021 at Olin College. The code for this lab has been adapted from:
https://github.com/ananya77041/baseball-elimination/blob/master/src/BaseballElimination.java'''

import sys
import math
import picos as pic
import networkx as nx
import itertools
import cvxopt
import matplotlib.pyplot as plt


class Division:
    '''
    The Division class represents a badminton division. This includes all the
    teams that are a part of that division, their winning and losing history,
    and their remaining games for the season.

    filename: name of a file with an input matrix that has info on teams &
    their games
    '''

    def __init__(self, filename):
        self.teams = {}
        self.G = nx.DiGraph()
        self.readDivision(filename)

    def readDivision(self, filename):
        '''Reads the information from the given file and builds up a dictionary
        of the teams that are a part of this division.

        filename: name of text file representing tournament outcomes so far
        & remaining games for each team
        '''
        f = open(filename, "r")
        lines = [line.split() for line in f.readlines()]
        f.close()

        lines = lines[1:]
        for ID, teaminfo in enumerate(lines):
            team = Team(int(ID), teaminfo[0], int(teaminfo[1]), int(teaminfo[2]), int(teaminfo[3]), list(map(int, teaminfo[4:])))
            self.teams[ID] = team

    def get_team_IDs(self):
        '''Gets the list of IDs that are associated with each of the teams
        in this division.

        return: list of IDs that are associated with each of the teams in the
        division
        '''
        return self.teams.keys()

    def is_eliminated(self, teamID, solver):
        '''Uses the given solver (either Linear Programming or Network Flows)
        to determine if the team with the given ID is mathematically
        eliminated from winning the division (aka winning more games than any
        other team) this season.

        teamID: ID of team that we want to check if it is eliminated
        solver: string representing whether to use the network flows or linear
        programming solver
        return: True if eliminated, False otherwise
        '''
        flag1 = False
        team = self.teams[teamID]

        temp = dict(self.teams)
        del temp[teamID]

        for _, other_team in temp.items():
            if team.wins + team.remaining < other_team.wins:
                flag1 = True

        saturated_edges = self.create_network(teamID)
        if not flag1:
            if solver == "Network Flows":
                flag1 = self.network_flows(saturated_edges)
            elif solver == "Linear Programming":
                flag1 = self.linear_programming(saturated_edges)

        return flag1

    def create_network(self, teamID, show=False):
        '''Builds up the network needed for solving the badminton elimination
        problem as a network flows problem & stores it in self.G. Returns a
        dictionary of saturated edges that maps team pairs to the amount of
        additional games they have against each other.

        teamID: ID of team that we want to check if it is eliminated
        return: dictionary of saturated edges that maps team pairs to
        the amount of additional games they have against each other
        '''

        saturated_edges = {}

        # Reset the graph
        self.G = nx.DiGraph()

        # Get dict of all other teams
        remaining_teams = dict(self.teams)
        remaining_teams.pop(teamID)

        # Populate saturated edges and first layer of G
        for pair in itertools.combinations(remaining_teams, 2):
            num = self.teams[pair[0]].get_against(pair[1])
            saturated_edges[pair] = num
            self.G.add_edge('S', pair, capacity=num)

        # Populate second layer of G
        for pair in saturated_edges:
            self.G.add_edge(pair, pair[0])
            self.G.add_edge(pair, pair[1])

        # Populate third layer of G
        for team in remaining_teams:
            self.G.add_edge(team, 'T', capacity=self.max_allowed(teamID, team))

        if show:
            pos = nx.spring_layout(self.G)
            labels = {}
            for edge in self.G.edges:
                if "capacity" in self.G.edges[edge[0], edge[1]]:
                    labels[edge] = self.G.edges[edge[0], edge[1]]["capacity"]
            nx.draw_networkx(self.G, pos=pos)
            nx.draw_networkx_edge_labels(self.G, pos, edge_labels=labels)
            plt.show()
        
        return saturated_edges

    def max_allowed(self, t1: int, t2: int) -> int:
        """Given two team id's t1 and t2, assume t1 wins all of their remaining
        games. Then, find out and return how many games t2 can win without
        displacing t1 from the top of the leaderboard.
        """
        team1 = self.teams[t1]
        team2 = self.teams[t2]
        return team1.wins + team1.remaining - team2.wins - 1

    def network_flows(self, saturated_edges):
        '''Uses network flows to determine if the team with given team ID
        has been eliminated. You can feel free to use the built in networkx
        maximum flow function or the maximum flow function you implemented as
        part of the in class implementation activity.

        saturated_edges: dictionary of saturated edges that maps team pairs to
        the amount of additional games they have against each other
        return: True if team is eliminated, False otherwise
        '''
        max_flow, flows = nx.algorithms.flow.maximum_flow(self.G, 'S', 'T')
        games_remaining = sum([x for x in saturated_edges.values()])
        print(f"max flow nf: {max_flow}")
        print(f"games remaining nf: {games_remaining}")

        return False if max_flow == games_remaining else True 

    def linear_programming(self, saturated_edges):
        '''Uses linear programming to determine if the team with given team ID
        has been eliminated. We recommend using a picos solver to solve the
        linear programming problem once you have it set up.
        Do not use the flow_constraint method that Picos provides (it does all of the work for you)
        We want you to set up the constraint equations using picos (hint: add_constraint is the method you want)

        saturated_edges: dictionary of saturated edges that maps team pairs to
        the amount of additional games they have against each other
        returns True if team is eliminated, False otherwise
        '''

        maxflow=pic.Problem()

        F = maxflow.add_variable('F')
        f = {}

        # get edge capacities
        c={}
        for e in self.G.edges():
            d = self.G[e[0]][e[1]]
            if "capacity" in d:
                if d["capacity"] < 0: 
                    # even if we win all games, still behind this team
                    # eliminate now, no need to solve
                    return True
                c[(e[0], e[1])]  = d["capacity"]
        cap=pic.new_param('c',c)

        # 0 <= flow <= capacity for each edge
        for e in self.G.edges():
            f[e] = maxflow.add_variable(f'f[{e}]')
            if "capacity" in self.G[e[0]][e[1]]:
                maxflow.add_constraint(f[e] <= cap[e])
            maxflow.add_constraint(f[e] >= 0)

        # flow_in == flow_out for each node; source_out == sink_in
        for n in self.G.adj:
            if n == 'S' or n == 'T':
                pass
            else:
                maxflow.add_constraint(
                    pic.sum([f[(x, n)] for x in self.G.predecessors(n)]) == 
                    pic.sum([f[(n, x)] for x in self.G.successors(n)])
                )

        # flow == flow out of source
        maxflow.add_constraint(
            F == pic.sum([f[('S', x)] for x in self.G.successors('S')])
        )

        # seems like this overconstrains the system sometimes...
        # flow out of source == flow into sink
        # maxflow.add_constraint(
        #     pic.sum([f[(x, 'T')] for x in self.G.predecessors('T')]) ==
        #     pic.sum([f[('S', x)] for x in self.G.successors('S')])
        # ) 

        # set objective and solve
        # print(maxflow)
        maxflow.set_objective("max", F)  
        maxflow.solve(solver="cvxopt")   
        
        games_remaining = sum([x for x in saturated_edges.values()])
        flow = float(F) # this is very dumb
        """
        An essay on overloading operators: Alex Wenstrup

        Never do this. Especially here and now. Please.
        """
        return not (abs(flow - games_remaining) < 0.1)


    def checkTeam(self, team):
        '''Checks that the team actually exists in this division.
        '''
        if team.ID not in self.get_team_IDs():
            raise ValueError("Team does not exist in given input.")

    def __str__(self):
        '''Returns pretty string representation of a division object.
        '''
        temp = ''
        for key in self.teams:
            temp = temp + f'{key}: {str(self.teams[key])} \n'
        return temp

class Team:
    '''
    The Team class represents one team within a badminton division for use in
    solving the badminton elimination problem. This class includes information
    on how many games the team has won and lost so far this season as well as
    information on what games they have left for the season.

    ID: ID to keep track of the given team
    teamname: human readable name associated with the team
    wins: number of games they have won so far
    losses: number of games they have lost so far
    remaining: number of games they have left this season
    against: dictionary that can tell us how many games they have left against
    each of the other teams
    '''

    def __init__(self, ID, teamname, wins, losses, remaining, against):
        self.ID = ID
        self.name = teamname
        self.wins = wins
        self.losses = losses
        self.remaining = remaining
        self.against = against

    def get_against(self, other_team=None):
        '''Returns number of games this team has against this other team.
        Raises an error if these teams don't play each other.
        '''
        try:
            num_games = self.against[other_team]
        except:
            raise ValueError("Team does not exist in given input.")

        return num_games

    def __str__(self):
        '''Returns pretty string representation of a team object.
        '''
        return f'{self.name} \t {self.wins} wins \t {self.losses} losses \t {self.remaining} remaining'

if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        division = Division(filename)
        for (ID, team) in division.teams.items():
            print(f'{team.name}: Eliminated? {division.is_eliminated(team.ID, "Linear Programming")}')
    else:
        print("To run this code, please specify an input file name. Example: python badminton_elimination.py teams2.txt.")
