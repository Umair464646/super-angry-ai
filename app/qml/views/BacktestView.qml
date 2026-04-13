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

        Label { text: "Backtest Lab"; color: "#E2EEFF"; font.pixelSize: 20; font.bold: true }
        ChartPanel { Layout.fillWidth: true; Layout.preferredHeight: 240; title: "Generation Fitness (proxy)"; series: appState.fitnessSeries; lineColor: "#2FB6FF" }

        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            radius: 12
            color: "#0F1725"
            border.color: "#1B2A41"
            ListView {
                anchors.fill: parent
                anchors.margins: 10
                model: appState.strategies
                delegate: Label {
                    required property var modelData
                    text: modelData.id + " | " + modelData.status + " | fit=" + modelData.fitness + " robust=" + modelData.robustness
                    color: "#9EC0E3"
                }
                ScrollBar.vertical: ScrollBar {}
            }
        }
    }
}
