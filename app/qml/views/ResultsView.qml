import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components"

Item {
    Layout.fillWidth: true
    Layout.fillHeight: true

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 10

        Label {
            text: "Results Studio"
            color: "#E1EEFF"
            font.pixelSize: 20
            font.bold: true
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            ChartPanel { Layout.fillWidth: true; Layout.fillHeight: true; title: "Leaderboard Fitness"; series: appState.fitnessSeries; lineColor: "#48BCFF" }
            ChartPanel { Layout.fillWidth: true; Layout.fillHeight: true; title: "Validation Quality"; series: appState.accuracySeries; lineColor: "#6BE0B4" }
        }
    }
}
