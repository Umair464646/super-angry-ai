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
            ChartPanel { Layout.fillWidth: true; Layout.preferredHeight: 260; title: "Generation Fitness Timeline"; series: appState.fitnessSeries; lineColor: "#46C3FF" }
            Rectangle {
                Layout.preferredWidth: 280
                Layout.fillHeight: true
                radius: 12
                color: "#111A29"
                border.color: "#1B2A41"
                Column {
                    anchors.fill: parent
                    anchors.margins: 10
                    spacing: 8
                    Label { text: "Evolution Diagnostics"; color: "#D9E9FF"; font.bold: true }
                    Label { text: "Diversity: dynamic"; color: "#A1C7EB" }
                    Label { text: "Exploration: adaptive"; color: "#A1C7EB" }
                    Label { text: "Mutation tiers: active"; color: "#A1C7EB" }
                    Label { text: "Stagnation guard: on"; color: "#A1C7EB" }
                }
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            radius: 12
            color: "#0F1725"
            border.color: "#1B2A41"
            ListView {
                anchors.fill: parent
                anchors.margins: 10
                model: appState.logs
                delegate: Label {
                    required property var modelData
                    text: modelData.msg
                    color: "#9EC0E3"
                }
                ScrollBar.vertical: ScrollBar {}
            }
        }
    }
}
