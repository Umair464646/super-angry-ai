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

        RowLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: 220
            ChartPanel { Layout.fillWidth: true; Layout.fillHeight: true; title: "Train Loss"; series: appState.lossSeries; lineColor: "#FF9B88" }
            ChartPanel { Layout.fillWidth: true; Layout.fillHeight: true; title: "Val Loss"; series: appState.valLossSeries; lineColor: "#FBD38D" }
            ChartPanel { Layout.fillWidth: true; Layout.fillHeight: true; title: "Train Accuracy"; series: appState.accuracySeries; lineColor: "#72E3BF" }
            ChartPanel { Layout.fillWidth: true; Layout.fillHeight: true; title: "Val Accuracy"; series: appState.valAccuracySeries; lineColor: "#57C9A8" }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 10

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
                    Label { text: "Regime Distribution"; color: "#D7EAFF"; font.bold: true }
                    Repeater {
                        model: Object.keys(appState.regimeCounts)
                        delegate: Label {
                            required property var modelData
                            text: modelData + ": " + appState.regimeCounts[modelData]
                            color: "#9BC2E6"
                        }
                    }
                }
            }

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
                    Repeater {
                        model: Object.keys(appState.featureImportance).slice(0, 12)
                        delegate: Label {
                            required property var modelData
                            text: modelData + ": " + Number(appState.featureImportance[modelData]).toFixed(4)
                            color: "#9BC2E6"
                        }
                    }
                }
            }
        }
    }
}
