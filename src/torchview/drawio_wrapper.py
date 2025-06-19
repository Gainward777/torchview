from __future__ import annotations

import graphviz
import torch
from torch import nn
from typing import Any

from .torchview import (
    process_input,
    forward_prop,
    validate_user_params,
    INPUT_DATA_TYPE,
    INPUT_SIZE_TYPE,
)
from .computation_graph import ComputationGraph
from .computation_node import TensorNode, ModuleNode, FunctionNode


class StyledComputationGraph(ComputationGraph):
    """ComputationGraph with a modern style for nodes and edges."""

    NODE_COLORS = {
        TensorNode: "#f8cecc",
        ModuleNode: "#dae8fc",
        FunctionNode: "#d5e8d4",
    }

    def add_node(self, node: TensorNode | ModuleNode | FunctionNode, subgraph: graphviz.Digraph | None = None) -> None:  # type: ignore[override]
        if node.node_id not in self.id_dict:
            self.id_dict[node.node_id] = self.running_node_id
            self.running_node_id += 1
        label = self.get_node_label(node)
        node_color = self.get_node_color(node)
        if subgraph is None:
            subgraph = self.visual_graph
        subgraph.node(
            name=f"{self.id_dict[node.node_id]}",
            label=label,
            fillcolor=node_color,
            color=node_color,
            shape="box",
            style="rounded,filled",
        )
        self.node_set.add(id(node))

    @staticmethod
    def get_node_color(node: TensorNode | ModuleNode | FunctionNode) -> str:  # type: ignore[override]
        return StyledComputationGraph.NODE_COLORS[type(node)]


def draw_graph_modern(
    model: nn.Module,
    input_data: INPUT_DATA_TYPE | None = None,
    input_size: INPUT_SIZE_TYPE | None = None,
    graph_name: str = "model",
    depth: int | float = 3,
    device: torch.device | str | None = None,
    dtypes: list[torch.dtype] | None = None,
    mode: str | None = None,
    strict: bool = True,
    expand_nested: bool = False,
    graph_dir: str | None = None,
    hide_module_functions: bool = True,
    hide_inner_tensors: bool = True,
    roll: bool = False,
    show_shapes: bool = True,
    save_graph: bool = False,
    filename: str | None = None,
    directory: str = ".",
    collect_attributes: bool = False,
    **kwargs: Any,
) -> StyledComputationGraph:
    """Generate a modern styled visualization of a PyTorch model.

    This wrapper mirrors :func:`torchview.draw_graph` but returns a graph with
    updated styling that resembles the diagram from the provided example.
    """

    if filename is None:
        filename = f"{graph_name}.gv"

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_mode = "eval" if mode is None else mode
    graph_dir = "TB" if graph_dir is None else graph_dir

    validate_user_params(model, input_data, input_size, depth, device, dtypes)

    graph_attr = {
        "ordering": "in",
        "rankdir": graph_dir,
    }
    node_attr = {
        "style": "rounded,filled",
        "shape": "box",
        "align": "center",
        "fontsize": "12",
        "ranksep": "0.2",
        "height": "0.3",
        "fontname": "Helvetica",
        "margin": "0.1",
    }
    edge_attr = {
        "fontsize": "10",
        "color": "#6c8ebf",
    }
    visual_graph = graphviz.Digraph(
        name=graph_name,
        engine="dot",
        strict=strict,
        graph_attr=graph_attr,
        node_attr=node_attr,
        edge_attr=edge_attr,
        directory=directory,
        filename=filename,
    )

    x, kwargs_record_tensor, input_nodes = process_input(
        input_data, input_size, kwargs, device, dtypes, collect_attributes
    )

    model_graph = StyledComputationGraph(
        visual_graph,
        input_nodes,
        show_shapes,
        expand_nested,
        hide_inner_tensors,
        hide_module_functions,
        roll,
        depth,
        collect_attributes,
    )

    forward_prop(model, x, device, model_graph, model_mode, **kwargs_record_tensor)

    model_graph.fill_visual_graph()

    if save_graph:
        model_graph.visual_graph.render(format="png")
    return model_graph
