import matplotlib.pyplot as plt

#used to graph accuracy/loss/etc. lines while training
class TrainGrapher:
    def __init__(self, should_graph, *graphs):
        self.should_graph = should_graph
        self.line_shapes = ['-b', '-r', '-g']

        if self.should_graph:
            self.fig, self.graphs = plt.subplots(1, len(graphs))

            if len(graphs) == 1:
                self.graphs = [self.graphs]
            self.graph_dict = {}

    # add lines you want to render at the beginning
    # returns list to add the values you want to graph
    def add_lines(self, graph, graph_loc, *lines):
        if not self.should_graph:
            return [[]] * len(lines)

        line_struct = []
        for i, title in enumerate(lines):
            shape = self.line_shapes[i % len(self.line_shapes)]
            line_struct.append((title, [], shape))

        self.graph_dict[graph] = (line_struct, graph_loc)

        return [i[1] for i in line_struct]

    # update the plot using the list returned from TrainGrapher::add_lines
    def update(self):
        if not self.should_graph:
            return

        # TODO: do i need to set_text every time?
        for i, graph_key in enumerate(self.graph_dict):
            self.graphs[i].cla()

            for line in self.graph_dict[graph_key][0]:
                self.graphs[i].plot([i + 1 for i in range(len(line[1]))], line[1], line[2], label = line[0])

            self.graphs[i].legend(loc = self.graph_dict[graph_key][1])
            self.graphs[i].title.set_text(graph_key)

        plt.xlabel("Epoch Number")
        plt.pause(0.0001)

    def show(self):
        if not self.should_graph:
            return
            
        plt.show()