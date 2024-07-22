from python_code.extensive_form_games.extensive_form_game import ExtensiveFormGame
from python_code.extensive_form_games.backward_induction import BackwardInduction
from python_code.utils.graph_visualisation import GraphVisualisation

game = AbstractExtensiveFormGame()
backward_induction = BackwardInduction(game)
solution = backward_induction.backward_induction(game.get_initial_state())

gv = GraphVisualisation(max_level = 5)
graph = gv.node_to_graph(game, game.game_tree(), print_value = False)
graph
