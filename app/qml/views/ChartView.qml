import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    Layout.fillWidth: true
    Layout.fillHeight: true

    property real lastX: 0

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 10

        RowLayout {
            Layout.fillWidth: true
            Label { text: "Chart Lab"; color: "#E2EEFF"; font.pixelSize: 20; font.bold: true }
            Item { Layout.fillWidth: true }
            ComboBox {
                id: tf
                model: ["1s", "1m", "5m", "15m"]
                currentIndex: Math.max(0, model.indexOf(appState.chartTimeframe))
                onActivated: appState.setChartTimeframe(currentText)
            }
            Button { text: "◀"; onClicked: appState.panChart(-Math.max(10, appState.chartWindowSize/5)) }
            Button { text: "▶"; onClicked: appState.panChart(Math.max(10, appState.chartWindowSize/5)) }
            Button { text: "-"; onClicked: appState.zoomChart(-1) }
            Button { text: "+"; onClicked: appState.zoomChart(1) }
            Label {
                text: "Window: " + appState.chartWindowSize + " candles"
                color: "#9BB6D8"
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            radius: 12
            color: "#0F1725"
            border.color: "#1B2A41"

            Canvas {
                id: candleCanvas
                anchors.fill: parent
                anchors.margins: 10

                onPaint: {
                    var ctx = getContext("2d")
                    ctx.reset()
                    ctx.fillStyle = "#0D1420"
                    ctx.fillRect(0, 0, width, height)

                    var candles = appState.chartCandles
                    if (!candles || candles.length === 0) {
                        ctx.fillStyle = "#7D93B0"
                        ctx.font = "13px sans-serif"
                        ctx.fillText("Load a dataset to render real candles", 16, 24)
                        return
                    }

                    var minL = candles[0].l
                    var maxH = candles[0].h
                    for (var i = 1; i < candles.length; ++i) {
                        minL = Math.min(minL, candles[i].l)
                        maxH = Math.max(maxH, candles[i].h)
                    }
                    var span = Math.max(1e-9, maxH - minL)
                    var cw = Math.max(2, width / candles.length)

                    for (var j = 0; j < candles.length; ++j) {
                        var c = candles[j]
                        var x = j * cw + cw * 0.5
                        var yH = height - ((c.h - minL) / span) * height
                        var yL = height - ((c.l - minL) / span) * height
                        var yO = height - ((c.o - minL) / span) * height
                        var yC = height - ((c.c - minL) / span) * height
                        var up = c.c >= c.o
                        ctx.strokeStyle = up ? "#3ED2A3" : "#F27D7D"
                        ctx.fillStyle = up ? "#3ED2A3" : "#F27D7D"
                        ctx.lineWidth = 1

                        ctx.beginPath()
                        ctx.moveTo(x, yH)
                        ctx.lineTo(x, yL)
                        ctx.stroke()

                        var top = Math.min(yO, yC)
                        var h = Math.max(1, Math.abs(yC - yO))
                        ctx.fillRect(x - cw * 0.32, top, cw * 0.64, h)
                    }
                }

                Connections {
                    target: appState
                    function onChartCandlesChanged() { candleCanvas.requestPaint() }
                }
                Component.onCompleted: requestPaint()
            }

            MouseArea {
                anchors.fill: parent
                acceptedButtons: Qt.LeftButton
                hoverEnabled: true
                onPressed: root.lastX = mouse.x
                onPositionChanged: {
                    if (!(mouse.buttons & Qt.LeftButton)) return
                    var dx = mouse.x - root.lastX
                    if (Math.abs(dx) > 10) {
                        var step = Math.round(dx / 10)
                        appState.panChart(-step)
                        root.lastX = mouse.x
                    }
                }
                onWheel: function(wheel) {
                    if (wheel.angleDelta.y > 0) appState.zoomChart(1)
                    else appState.zoomChart(-1)
                }
            }
        }
    }
}
