from dash import Dash, html, dcc, ctx, callback, Output, Input
import dash_bootstrap_components as dbc
from llama_cpp import Llama

llm = Llama(
    model_path="/Users/yasas/Applications/llama.cpp/models/llama-2-13b-chat/ggml-model-q4_0.gguf"
)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, "./static/base.css"])

app.layout = html.Div(
    [
        dcc.Interval(
            id="interval-component",
            interval=1 * 100,
            n_intervals=0,  # in milliseconds
        ),
        dbc.InputGroup(
            [
                dbc.Textarea(
                    id="input-prompt-text",
                    placeholder="Prompt",
                    style={"height": 56},
                ),
                dbc.Button("Ask!", id="input-ask-btn", n_clicks=0),
            ]
        ),
        html.Div(
            id="output-container-ctx",
            style={
                "height": "100%",
            },
        ),
    ],
    style={
        "padding": "50px",
        "width": "100%",
        "height": "100%",
    },
)

global_value = {}


@callback(
    Output("output-container-ctx", "children"),
    Input("input-prompt-text", "value"),
    Input("input-ask-btn", "n_clicks"),
    Input("interval-component", "n_intervals"),
)
def display(prompt, _1, _2):
    global global_value
    if not prompt:
        return html.Div()
    stream = global_value.pop("stream", None)
    output_text = global_value.get("output_text", "You: ")
    if ctx.triggered_id in {"input-ask-btn", "interval-component"}:
        if ctx.triggered_id == "input-ask-btn":
            output_text += " [NEWLINE] "
            stream = llm(
                f"Q: {output_text + 'User: ' + prompt} A: ",
                max_tokens=4000,
                stop=["Q:", "User:"],
                echo=True,
                stream=True,
            )
        if stream is not None:
            try:
                out = next(stream)
                output_text += out["choices"][0]["text"]
            except StopIteration:
                stream = None
    global_value["stream"] = stream
    global_value["output_text"] = output_text
    return html.Div([html.P(text) for text in output_text.split(" [NEWLINE] ")])


if __name__ == "__main__":
    app.run(debug=True)
