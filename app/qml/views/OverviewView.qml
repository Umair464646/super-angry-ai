import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../components"

Item {
    Layout.fillWidth: true
    Layout.fillHeight: true
    ScrollView {
        anchors.fill: parent
        contentWidth: availableWidth

        ColumnLayout {
            width: parent.width
            spacing: 10

            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 120
                radius: 12
                color: "#111C2D"
                border.color: "#1A2A40"
                Column {
                    anchors.fill: parent
                    anchors.margins: 14
                    spacing: 8
                    Label { text: "Research Overview"; color: "#E2EEFF"; font.pixelSize: 18; font.bold: true }
                    Label { text: "Live AI pipeline status, generation health, and portfolio-ready strategy ranking."; color: "#9BB6D8" }
                    Label { text: "Model status: " + appState.modelStatus; color: "#8FD3FF" }
                }
            }

            RowLayout {
                Layout.fillWidth: true
                Layout.preferredHeight: 220
                ChartPanel { Layout.fillWidth: true; Layout.fillHeight: true; title: "Best Fitness"; series: appState.fitnessSeries; lineColor: "#2FB6FF" }
                ChartPanel { Layout.fillWidth: true; Layout.fillHeight: true; title: "Loss"; series: appState.lossSeries; lineColor: "#FE8A79" }
                ChartPanel { Layout.fillWidth: true; Layout.fillHeight: true; title: "Accuracy"; series: appState.accuracySeries; lineColor: "#61E5B6" }
            }
        }
    }
}
