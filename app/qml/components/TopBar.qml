import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: root
    color: "#0E1420"
    radius: 12
    border.color: "#1A2638"
    implicitHeight: 70

    signal startClicked()
    signal pauseClicked()
    signal stopClicked()

    RowLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 10

        ComboBox { model: ["Project Alpha"]; Layout.preferredWidth: 130 }

        TextField {
            id: datasetPath
            Layout.preferredWidth: 360
            Layout.fillWidth: true
            placeholderText: "Dataset path (CSV/Parquet)"
            text: appState.datasetPath
            onEditingFinished: appState.setDatasetPath(text)
        }

        ComboBox { model: ["1m", "5m", "15m", "1h"]; currentIndex: 0; Layout.preferredWidth: 90 }

        Button { text: "Start"; onClicked: { appState.setDatasetPath(datasetPath.text); root.startClicked() } }
        Button { text: "Pause"; onClicked: root.pauseClicked() }
        Button { text: "Stop"; onClicked: root.stopClicked() }

        Rectangle {
            radius: 10
            color: "#132238"
            border.color: "#2E4B72"
            Layout.preferredHeight: 36
            Layout.preferredWidth: 240
            Label {
                anchors.centerIn: parent
                text: "Stage: " + appState.stageText
                color: "#BEE0FF"
                font.pixelSize: 12
            }
        }

        Rectangle {
            radius: 10
            color: "#132238"
            border.color: "#2E4B72"
            Layout.preferredHeight: 36
            Layout.preferredWidth: 145
            Label {
                anchors.centerIn: parent
                text: "Model: " + appState.modelStatus
                color: "#BEE0FF"
                font.pixelSize: 12
            }
        }
    }
}
