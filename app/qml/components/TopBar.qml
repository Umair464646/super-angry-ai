import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: root
    color: "#0E1420"
    radius: 12
    border.color: "#1A2638"
    implicitHeight: 64

    signal startClicked()
    signal pauseClicked()
    signal stopClicked()

    RowLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 10

        ComboBox { model: ["Project Alpha"]; Layout.preferredWidth: 140 }
        ComboBox { model: ["ETHUSDT_1s_refined.parquet"]; Layout.preferredWidth: 240 }
        ComboBox { model: ["1m", "5m", "15m", "1h"]; currentIndex: 0; Layout.preferredWidth: 90 }

        Item { Layout.preferredWidth: 6 }

        Button { text: "Start"; onClicked: root.startClicked() }
        Button { text: "Pause"; onClicked: root.pauseClicked() }
        Button { text: "Stop"; onClicked: root.stopClicked() }

        Item { Layout.fillWidth: true }

        Rectangle {
            radius: 10
            color: "#132238"
            border.color: "#2E4B72"
            Layout.preferredHeight: 34
            Layout.preferredWidth: 170
            Label {
                anchors.centerIn: parent
                text: "Model: " + appState.modelStatus
                color: "#BEE0FF"
                font.pixelSize: 12
            }
        }
    }
}
