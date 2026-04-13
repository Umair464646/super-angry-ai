import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: root
    property string title: "Chart"
    property var series: []
    property color lineColor: "#39B8FF"

    radius: 12
    color: "#101827"
    border.color: "#1A2A40"

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 8

        Label { text: root.title; color: "#D5E7FF"; font.bold: true }

        Canvas {
            id: canvas
            Layout.fillWidth: true
            Layout.fillHeight: true
            onPaint: {
                var ctx = getContext("2d")
                ctx.reset()
                ctx.fillStyle = "#0D1420"
                ctx.fillRect(0, 0, width, height)

                if (!root.series || root.series.length < 2) {
                    ctx.fillStyle = "#6E819D"
                    ctx.font = "12px sans-serif"
                    ctx.fillText("Waiting for live data", 12, 20)
                    return
                }

                var minV = root.series[0]
                var maxV = root.series[0]
                for (var i = 1; i < root.series.length; ++i) {
                    minV = Math.min(minV, root.series[i])
                    maxV = Math.max(maxV, root.series[i])
                }
                var span = Math.max(0.0001, maxV - minV)
                ctx.strokeStyle = root.lineColor
                ctx.lineWidth = 2
                ctx.beginPath()
                for (var j = 0; j < root.series.length; ++j) {
                    var x = (j / (root.series.length - 1)) * width
                    var y = height - ((root.series[j] - minV) / span) * height
                    if (j === 0) ctx.moveTo(x, y)
                    else ctx.lineTo(x, y)
                }
                ctx.stroke()
            }
            Connections {
                target: root
                function onSeriesChanged() { canvas.requestPaint() }
            }
            Component.onCompleted: requestPaint()
        }
    }
}
