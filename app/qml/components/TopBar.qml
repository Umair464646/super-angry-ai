import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

Rectangle {
    id: root
    color: "#0E1420"
    radius: 12
    border.color: "#1A2638"
    implicitHeight: 70

    signal startClicked()
    signal loadClicked()
    signal pauseClicked()
    signal stopClicked()

    FileDialog {
        id: fileDialog
        title: "Select market dataset"
        nameFilters: ["Data files (*.parquet *.pq *.csv)", "All files (*)"]
        onAccepted: {
            datasetPath.text = selectedFile.toString()
            appState.logUiEvent("File selected from dialog")
            appState.setDatasetPath(datasetPath.text)
        }
    }

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

        Button { text: "Browse"; onClicked: { appState.logUiEvent("Browse clicked"); fileDialog.open() } }
        ComboBox { model: ["1m", "5m", "15m", "1h"]; currentIndex: 0; Layout.preferredWidth: 90 }

        Button { text: "Load"; onClicked: { appState.logUiEvent("Load clicked"); appState.setDatasetPath(datasetPath.text); root.loadClicked() } }
        Button { text: "Start"; onClicked: { appState.logUiEvent("Start clicked"); appState.setDatasetPath(datasetPath.text); root.startClicked() } }
        Button { text: "Pause"; onClicked: { appState.logUiEvent("Pause clicked"); root.pauseClicked() } }
        Button { text: "Stop"; onClicked: { appState.logUiEvent("Stop clicked"); root.stopClicked() } }

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
