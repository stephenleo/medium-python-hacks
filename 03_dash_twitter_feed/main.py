import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Setup the Dash App
external_stylesheets = [dbc.themes.LITERA]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Server
server = app.server

# App Layout
app.layout = html.Table(
    [
        html.Tr(
            [
                html.H1(html.Center(html.B("View Twitter Feed"))),
                html.Br(),
                html.Center(
                    [
                        html.Div(
                            dbc.Input(
                                id="tweet_id_list",
                                value="",
                                placeholder="Enter tweet ids separated by comma",
                                style={"width": "700px"},
                            )
                        ),
                        html.Div(id="blank-output"),
                        dcc.Store(id="tweet_ids"),
                        html.Div(id="tweet_feed"),
                    ]
                ),
            ]
        )
    ],
    style={"marginLeft": "auto", "marginRight": "auto"},
)

# Callbacks
@app.callback(
    Output("tweet_feed", "children"),
    Output("tweet_ids", "data"),
    Input("tweet_id_list", "value"),
)
def tweet_feed(tweet_id_list):
    tweet_ids = [tweet.strip() for tweet in tweet_id_list.split(",")]
    print(tweet_ids)
    return ([html.Div(id=f"container_{tweet_id}") for tweet_id in tweet_ids], tweet_ids)

app.clientside_callback(
    """
    function(tweet_ids) {
        console.log(tweet_ids)
        for (let i = 0; i < tweet_ids.length; i++) {
            twttr.widgets.createTweet(tweet_ids[i], document.getElementById('container_' + tweet_ids[i]));
        }
    }
    """,
    Output("blank-output", "children"),
    Input("tweet_ids", "data"),
)


if __name__ == "__main__":
    app.run_server(debug=True)
