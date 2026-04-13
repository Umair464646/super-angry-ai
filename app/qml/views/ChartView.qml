import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    Layout.fillWidth: true
    Layout.fillHeight: true

    property real lastX: 0
    property real hoverX: -1
    property real hoverY: -1
    property int hoverIndex: -1

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 8

        RowLayout {
            Layout.fillWidth: true
            Label { text: "Chart Lab"; color: "#E2EEFF"; font.pixelSize: 20; font.bold: true }
            Item { Layout.fillWidth: true }
            ComboBox {
                id: tf
                model: ["1s", "1m", "5m", "15m", "30m", "1h", "2h", "4h"]
                currentIndex: Math.max(0, model.indexOf(appState.chartTimeframe))
                onActivated: appState.setChartTimeframe(currentText)
            }
            Button { text: "◀"; onClicked: appState.panChart(-Math.max(10, appState.chartWindowSize/5)) }
            Button { text: "▶"; onClicked: appState.panChart(Math.max(10, appState.chartWindowSize/5)) }
            Button { text: "-"; onClicked: appState.zoomChart(-1) }
            Button { text: "+"; onClicked: appState.zoomChart(1) }
            Label { text: "Window: " + appState.chartWindowSize + " candles"; color: "#9BB6D8" }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            radius: 12
            color: "#0B1020"
            border.color: "#1B2A41"

            Canvas {
                id: candleCanvas
                anchors.fill: parent
                anchors.margins: 8

                onPaint: {
                    var ctx = getContext("2d")
                    ctx.reset()
                    ctx.fillStyle = "#0B1020"
                    ctx.fillRect(0, 0, width, height)

                    var candles = appState.chartCandles
                    if (!candles || candles.length === 0) {
                        ctx.fillStyle = "#7D93B0"
                        ctx.font = "13px sans-serif"
                        ctx.fillText("Load a dataset to render real candles", 16, 24)
                        return
                    }

                    var rightAxisW = 84
                    var bottomAxisH = 28
                    var plotW = width - rightAxisW
                    var plotH = height - bottomAxisH

                    var minL = candles[0].l
                    var maxH = candles[0].h
                    for (var i = 1; i < candles.length; ++i) {
                        minL = Math.min(minL, candles[i].l)
                        maxH = Math.max(maxH, candles[i].h)
                    }
                    var pad = (maxH - minL) * 0.08
                    maxH += pad
                    minL -= pad
                    var span = Math.max(1e-9, maxH - minL)
                    var cw = Math.max(3, plotW / candles.length)

                    // grid
                    ctx.strokeStyle = "rgba(86,105,146,0.22)"
                    ctx.lineWidth = 1
                    for (var gy = 0; gy <= 8; ++gy) {
                        var y = (gy / 8) * plotH
                        ctx.beginPath()
                        ctx.moveTo(0, y)
                        ctx.lineTo(plotW, y)
                        ctx.stroke()
                    }
                    for (var gx = 0; gx <= 10; ++gx) {
                        var xg = (gx / 10) * plotW
                        ctx.beginPath()
                        ctx.moveTo(xg, 0)
                        ctx.lineTo(xg, plotH)
                        ctx.stroke()
                    }

                    // candles
                    for (var j = 0; j < candles.length; ++j) {
                        var c = candles[j]
                        var x = j * cw + cw * 0.5
                        var yH = plotH - ((c.h - minL) / span) * plotH
                        var yL = plotH - ((c.l - minL) / span) * plotH
                        var yO = plotH - ((c.o - minL) / span) * plotH
                        var yC = plotH - ((c.c - minL) / span) * plotH
                        var up = c.c >= c.o
                        ctx.strokeStyle = up ? "#22ab94" : "#f23645"
                        ctx.fillStyle = up ? "#22ab94" : "#f23645"
                        ctx.lineWidth = 1

                        ctx.beginPath()
                        ctx.moveTo(x, yH)
                        ctx.lineTo(x, yL)
                        ctx.stroke()

                        var top = Math.min(yO, yC)
                        var h = Math.max(1, Math.abs(yC - yO))
                        ctx.fillRect(x - cw * 0.34, top, cw * 0.68, h)
                    }

                    // right y-axis labels
                    ctx.fillStyle = "#9FB5D7"
                    ctx.font = "11px sans-serif"
                    for (var py = 0; py <= 8; ++py) {
                        var yy = (py / 8) * plotH
                        var price = maxH - (py / 8) * (maxH - minL)
                        ctx.fillText(price.toFixed(2), plotW + 8, yy + 4)
                    }

                    // bottom time labels
                    var tickCount = Math.min(8, candles.length - 1)
                    for (var tx = 0; tx <= tickCount; ++tx) {
                        var idx = Math.round((tx / tickCount) * (candles.length - 1))
                        var xx = (idx + 0.5) * cw
                        var t = String(candles[idx].t)
                        ctx.fillStyle = "#8FA8C8"
                        ctx.fillText(t.substring(5, 16), Math.max(2, xx - 32), plotH + 16)
                    }

                    // crosshair + ohlc readout
                    if (hoverX >= 0 && hoverY >= 0 && hoverX <= plotW && hoverY <= plotH) {
                        var hi = Math.max(0, Math.min(candles.length - 1, Math.floor(hoverX / cw)))
                        hoverIndex = hi
                        var hc = candles[hi]
                        var hx = (hi + 0.5) * cw
                        ctx.strokeStyle = "rgba(220,230,255,0.45)"
                        ctx.setLineDash([4,4])
                        ctx.beginPath(); ctx.moveTo(hx, 0); ctx.lineTo(hx, plotH); ctx.stroke()
                        ctx.beginPath(); ctx.moveTo(0, hoverY); ctx.lineTo(plotW, hoverY); ctx.stroke()
                        ctx.setLineDash([])

                        ctx.fillStyle = "#dbe7ff"
                        ctx.font = "12px sans-serif"
                        var txt = "O " + hc.o.toFixed(2) + "  H " + hc.h.toFixed(2) + "  L " + hc.l.toFixed(2) + "  C " + hc.c.toFixed(2)
                        ctx.fillText(txt, 10, 14)
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
                onPressed: {
                    root.lastX = mouse.x
                    root.hoverX = mouse.x - 8
                    root.hoverY = mouse.y - 8
                    candleCanvas.requestPaint()
                }
                onPositionChanged: {
                    root.hoverX = mouse.x - 8
                    root.hoverY = mouse.y - 8
                    if (mouse.buttons & Qt.LeftButton) {
                        var dx = mouse.x - root.lastX
                        if (Math.abs(dx) > 6) {
                            var step = Math.round(dx / 6)
                            appState.panChart(-step)
                            root.lastX = mouse.x
                        }
                    }
                    candleCanvas.requestPaint()
                }
                onExited: {
                    root.hoverX = -1
                    root.hoverY = -1
                    candleCanvas.requestPaint()
                }
                onWheel: function(wheel) {
                    if (wheel.angleDelta.y > 0) appState.zoomChart(1)
                    else appState.zoomChart(-1)
                }
            }
        }
    }
}
