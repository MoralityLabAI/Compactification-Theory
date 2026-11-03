pragma circom 2.1.6;
// Generated from Algernon-02 pointer-machine semantics, N=4, STEPS=8.
// Signal arrays are public only through main's public annotation.
template PointerTrace(N, STEPS) {
    signal input edge[N][N];
    signal input start[N];
    signal state[STEPS + 1][N];
    signal product[STEPS][N][N];
    signal output final[N];

    var startSum = 0;
    for (var i = 0; i < N; i++) {
        start[i] * (start[i] - 1) === 0;
        state[0][i] <== start[i];
        startSum += start[i];
        var edgeSum = 0;
        for (var j = 0; j < N; j++) {
            edge[i][j] * (edge[i][j] - 1) === 0;
            edgeSum += edge[i][j];
        }
        edgeSum === 1;
    }
    startSum === 1;

    for (var t = 0; t < STEPS; t++) {
        for (var i = 0; i < N; i++) {
            for (var j = 0; j < N; j++) {
                product[t][i][j] <== state[t][i] * edge[i][j];
            }
        }
        for (var j = 0; j < N; j++) {
            var nextSum = 0;
            for (var i = 0; i < N; i++) {
                nextSum += product[t][i][j];
            }
            state[t + 1][j] <== nextSum;
        }
    }
    for (var t = 0; t <= STEPS; t++) {
        var stateSum = 0;
        for (var i = 0; i < N; i++) {
            state[t][i] * (state[t][i] - 1) === 0;
            stateSum += state[t][i];
        }
        stateSum === 1;
    }
    for (var i = 0; i < N; i++) {
        final[i] <== state[STEPS][i];
    }
}
component main {public [edge, start]} = PointerTrace(4, 8);
