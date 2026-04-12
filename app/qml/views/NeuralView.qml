import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components"

Item {
    Layout.fillWidth: true
    Layout.fillHeight: true
    property real pulse: 0

    Timer {
        interval: 120
        running: appState.modelStatus === "running"
        repeat: true
        onTriggered: {
            pulse = (pulse + 0.08) % 6.28
            netCanvas.requestPaint()
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 10

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 10

            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                radius: 12
                color: "#0F1725"
                border.color: "#1B2A41"

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 10
                    Label { text: "Neural Network Live Topology"; color: "#DBEBFF"; font.bold: true }
                    Canvas {
                        id: netCanvas
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        onPaint: {
                            var ctx = getContext("2d")
                            ctx.reset()
                            ctx.fillStyle = "#0D1420"
                            ctx.fillRect(0, 0, width, height)

                            var layers = [5, 8, 6, 3]
                            var layerX = []
                            for (var l = 0; l < layers.length; ++l)
                                layerX.push((l + 1) * width / (layers.length + 1))

                            // edges
                            for (var a = 0; a < layers.length - 1; ++a) {
                                for (var i = 0; i < layers[a]; ++i) {
                                    for (var j = 0; j < layers[a+1]; ++j) {
                                        var y1 = (i + 1) * height / (layers[a] + 1)
                                        var y2 = (j + 1) * height / (layers[a+1] + 1)
                                        var strength = Math.abs(Math.sin(pulse + i*0.31 + j*0.17 + a*0.7))
                                        ctx.strokeStyle = "rgba(62, 171, 255," + (0.1 + strength * 0.45) + ")"
                                        ctx.lineWidth = 0.5 + strength * 1.6
                                        ctx.beginPath()
                                        ctx.moveTo(layerX[a], y1)
                                        ctx.lineTo(layerX[a+1], y2)
                                        ctx.stroke()
                                    }
                                }
                            }

                            // nodes
                            for (var k = 0; k < layers.length; ++k) {
                                for (var n = 0; n < layers[k]; ++n) {
                                    var y = (n + 1) * height / (layers[k] + 1)
                                    var act = Math.abs(Math.sin(pulse + k * 0.9 + n * 0.45))
                                    var r = 5 + act * 6
                                    ctx.fillStyle = "rgba(107, 220, 255," + (0.35 + act * 0.6) + ")"
                                    if (k === layers.length - 1 && n === 0)
                                        ctx.fillStyle = "rgba(107,255,183," + (0.35 + act * 0.6) + ")"
                                    if (k === layers.length - 1 && n === 1)
                                        ctx.fillStyle = "rgba(255,160,108," + (0.35 + act * 0.6) + ")"
                                    if (k === layers.length - 1 && n === 2)
                                        ctx.fillStyle = "rgba(163,174,192," + (0.35 + act * 0.6) + ")"
                                    ctx.beginPath()
                                    ctx.arc(layerX[k], y, r, 0, 6.28)
                                    ctx.fill()
                                }
                            }
                        }
                    }
                }
            }

            ColumnLayout {
                Layout.preferredWidth: 360
                Layout.fillHeight: true
                spacing: 10
                ChartPanel { Layout.fillWidth: true; Layout.preferredHeight: 170; title: "Train Loss"; series: appState.lossSeries; lineColor: "#FF9B88" }
                ChartPanel { Layout.fillWidth: true; Layout.preferredHeight: 170; title: "Train Accuracy"; series: appState.accuracySeries; lineColor: "#72E3BF" }
                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    radius: 12
                    color: "#101A2A"
                    border.color: "#1B2A41"
                    Column {
                        anchors.fill: parent
                        anchors.margins: 10
                        spacing: 8
                        Label { text: "Feature Importance"; color: "#D7EAFF"; font.bold: true }
                        Label { text: "• VWAP distance"; color: "#9BC2E6" }
                        Label { text: "• Delta imbalance"; color: "#9BC2E6" }
                        Label { text: "• Volatility expansion"; color: "#9BC2E6" }
                        Label { text: "• Trend slope"; color: "#9BC2E6" }
                    }
                }
            }
        }
    }
}
