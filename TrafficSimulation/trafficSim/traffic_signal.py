import random

class TrafficSignal:
    def __init__(self, roads, config={}):
        # Initialize roads
        self.roads = roads
        # Set default configuration
        self.set_default_config()

        # Update configuration
        for attr, val in config.items():
            setattr(self, attr, val)
        # Calculate properties
        self.init_properties()

    def set_default_config(self):
        self.cycle = [(False, False, False, True), (False, False, True, False), (False, True, False, False), (True, False, False, False)]
        self.slow_distance = 50
        self.slow_factor = 0.4
        self.stop_distance = 15

        self.current_cycle_index = 4

        self.last_t = 0

    def init_properties(self):
        for i in range(len(self.roads)):
            for road in self.roads[i]:
                road.set_traffic_signal(self, i)

    @property
    def current_cycle(self):
        return self.cycle[self.current_cycle_index % 4]
    
    def update(self, sim):
        # cycle_length = 45
        # k = (sim.t // cycle_length) % 4
        # self.current_cycle_index = self.current_cycle_index + int(k)
        # self.current_cycle_index = self.current_cycle_index % 4
        if(len(self.roads) < 4):
            self.current_cycle_index = 3

    def update_state(self):
        tmp = self.current_cycle_index + 1
        self.current_cycle_index = tmp % 4
